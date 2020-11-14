from typing import Optional, Type
from habitat import Config, Dataset
from utils.vis_utils import observations_to_image, append_text_to_image
import cv2
from gym.spaces.dict_space import Dict as SpaceDict
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from habitat.core.spaces import ActionSpace, EmptySpace
import numpy as np
from env_utils.habitat_env import RLEnv, MIN_DIST, MAX_DIST
import habitat
from habitat.utils.visualizations.utils import images_to_video
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
import env_utils.noisy_actions
from env_utils.noisy_actions import CustomActionSpaceConfiguration
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat_sim.utils.common import quat_to_coeffs
RENDER = True
NOISY = True
class VisTargetNavEnv(RLEnv):
    metadata = {'render.modes': ['rgb_array']}
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self.noise = NOISY
        print('[VisTargetNavEnv] NOISY ACTUATION : ', self.noise)
        if hasattr(config,'AGENT_TASK'):
            self.agent_task = config.AGENT_TASK
        else:
            self.agent_task = 'search'
        if self.agent_task == 'homing':
            self.num_goals = 2
        else:
            self.num_goals = config.NUM_GOALS
            
            
        task_config = config.TASK_CONFIG
        task_config.defrost()
        #task_config.TASK.TOP_DOWN_MAP.MAP_RESOLUTION = 1250
        task_config.TASK.TOP_DOWN_MAP.DRAW_SOURCE = True
        task_config.TASK.TOP_DOWN_MAP.DRAW_SHORTEST_PATH = True
        task_config.TASK.TOP_DOWN_MAP.FOG_OF_WAR.VISIBILITY_DIST = 2.0
        task_config.TASK.TOP_DOWN_MAP.FOG_OF_WAR.FOV = 360
        task_config.TASK.TOP_DOWN_MAP.FOG_OF_WAR.DRAW = True
        task_config.TASK.TOP_DOWN_MAP.DRAW_VIEW_POINTS = False
        task_config.TASK.TOP_DOWN_MAP.DRAW_GOAL_POSITIONS = True
        task_config.TASK.TOP_DOWN_MAP.DRAW_GOAL_AABBS = False
        if ('GMT' in config.POLICY or 'NTS' in config.POLICY) and RENDER:
            task_config.TASK.TOP_DOWN_GRAPH_MAP = config.TASK_CONFIG.TASK.TOP_DOWN_MAP.clone()
            if 'GMT' in config.POLICY:
                task_config.TASK.TOP_DOWN_GRAPH_MAP.TYPE = "TopDownGraphMap"
            elif 'NTS' in config.POLICY:
                task_config.TASK.TOP_DOWN_GRAPH_MAP.TYPE = 'NTSGraphMap'
                task_config.TASK.TOP_DOWN_GRAPH_MAP.MAP_RESOLUTION = 4000
            task_config.TASK.TOP_DOWN_GRAPH_MAP.NUM_TOPDOWN_MAP_SAMPLE_POINTS = 20000
            task_config.TASK.MEASUREMENTS += ['TOP_DOWN_GRAPH_MAP']
            if 'TOP_DOWN_MAP' in config.TASK_CONFIG.TASK.MEASUREMENTS:
                task_config.TASK.MEASUREMENTS = [k for k in task_config.TASK.MEASUREMENTS if
                                                        'TOP_DOWN_MAP' != k]
        task_config.SIMULATOR.ACTION_SPACE_CONFIG = "CustomActionSpaceConfiguration"
        task_config.TASK.POSSIBLE_ACTIONS = task_config.TASK.POSSIBLE_ACTIONS + ['NOISY_FORWARD', 'NOISY_RIGHT', 'NOISY_LEFT']
        task_config.TASK.ACTIONS.NOISY_FORWARD = habitat.config.Config()
        task_config.TASK.ACTIONS.NOISY_FORWARD.TYPE = "NOISYFORWARD"
        task_config.TASK.ACTIONS.NOISY_RIGHT = habitat.config.Config()
        task_config.TASK.ACTIONS.NOISY_RIGHT.TYPE = "NOISYRIGHT"
        task_config.TASK.ACTIONS.NOISY_LEFT = habitat.config.Config()
        task_config.TASK.ACTIONS.NOISY_LEFT.TYPE = "NOISYLEFT"
        task_config.TASK.MEASUREMENTS = ['GOAL_INDEX'] + task_config.TASK.MEASUREMENTS + ['SOFT_SPL']
        task_config.TASK.DISTANCE_TO_GOAL.TYPE = 'Custom_DistanceToGoal'
        if self.agent_task != 'search':
            task_config.TASK.SPL.TYPE = 'Custom_SPL'
        task_config.TASK.SOFT_SPL.TYPE = 'Custom_SoftSPL'
        task_config.TASK.GOAL_INDEX = task_config.TASK.SPL.clone()
        task_config.TASK.GOAL_INDEX.TYPE = 'GoalIndex'
        task_config.freeze()
        self.config = config
        self._core_env_config = config.TASK_CONFIG
        self._reward_measure_name = config.REWARD_METHOD
        self._success_measure_name = config.SUCCESS_MEASURE
        self.success_distance = config.SUCCESS_DISTANCE
        self._previous_measure = None
        self._previous_action = -1
        self.time_t = 0
        self.stuck = 0
        self.follower = None
        if 'NOISY_FORWARD' not in HabitatSimActions:
            HabitatSimActions.extend_action_space("NOISY_FORWARD")
            HabitatSimActions.extend_action_space("NOISY_RIGHT")
            HabitatSimActions.extend_action_space("NOISY_LEFT")
        if 'STOP' in task_config.TASK.POSSIBLE_ACTIONS:
            self.action_dict = {0: HabitatSimActions.STOP,
                           1: "NOISY_FORWARD",
                           2: "NOISY_LEFT",
                           3: "NOISY_RIGHT"}
        else:
            self.action_dict = {0: "NOISY_FORWARD",
                           1: "NOISY_LEFT",
                           2: "NOISY_RIGHT"}
        super().__init__(self._core_env_config, dataset)
        act_dict = {"MOVE_FORWARD": EmptySpace(),
                    'TURN_LEFT': EmptySpace(),
                    'TURN_RIGHT': EmptySpace()
        }
        if 'STOP' in task_config.TASK.POSSIBLE_ACTIONS:
            act_dict.update({'STOP': EmptySpace()})
        self.action_space = ActionSpace(act_dict)
        obs_dict = {
                'panoramic_rgb': self.habitat_env._task.sensor_suite.observation_spaces.spaces['panoramic_rgb'],
                'panoramic_depth': self.habitat_env._task.sensor_suite.observation_spaces.spaces['panoramic_depth'],
                'target_goal': self.habitat_env._task.sensor_suite.observation_spaces.spaces['target_goal'],
                'step': Box(low=np.array(0),high=np.array(500), dtype=np.float32),
                'prev_act': Box(low=np.array(-1), high=np.array(self.action_space.n), dtype=np.int32),
                'gt_action': Box(low=np.array(-1), high=np.array(self.action_space.n), dtype=np.int32),
                'position': Box(low=-np.Inf, high=np.Inf, shape=(3,), dtype=np.float32),
                'target_pose': Box(low=-np.Inf, high=np.Inf, shape=(3,), dtype=np.float32),
                'distance': Box(low=-np.Inf, high=np.Inf, shape=(1,), dtype=np.float32),
            }
        if 'GMT' in config.POLICY and RENDER:
            self.mapper = self.habitat_env.task.measurements.measures['top_down_map']
            #obs_dict.update({'unexplored':Box(low=0, high=1, shape=(self.mapper.delta,), dtype=np.int32),
            #    'neighbors': Box(low=0, high=1, shape=(self.mapper.delta,), dtype=np.int32),})
        else:
            self.mapper = None
        if 'aux' in self.config.POLICY:
            self.return_have_been = True
            self.return_target_dist_score = True
            obs_dict.update({'have_been': Box(low=0, high=1, shape=(1,), dtype=np.int32),
                             'target_dist_score': Box(low=0, high=1, shape=(1,), dtype=np.float32),
                             })
        else:
            self.return_have_been = False
            self.return_target_dist_score = False

        self.observation_space = SpaceDict(obs_dict)

        
        if config.DIFFICULTY == 'easy':
            self.habitat_env.difficulty = 'easy'
            self.habitat_env.MIN_DIST, self.habitat_env.MAX_DIST = 1.5, 3.0
        elif config.DIFFICULTY == 'medium':
            self.habitat_env.difficulty = 'medium'
            self.habitat_env.MIN_DIST, self.habitat_env.MAX_DIST = 3.0, 5.0
        elif config.DIFFICULTY == 'hard':
            self.habitat_env.difficulty = 'hard'
            self.habitat_env.MIN_DIST, self.habitat_env.MAX_DIST = 5.0, 10.0
        elif config.DIFFICULTY == 'random':
            self.habitat_env.difficulty = 'random'
            self.habitat_env.MIN_DIST, self.habitat_env.MAX_DIST = 3.0, 10.0
        else:
            raise NotImplementedError


        self.habitat_env._num_goals = self.num_goals
        self.habitat_env._agent_task = self.agent_task
        print('current task : %s'%(self.agent_task))
        print('current difficulty %s, MIN_DIST %f, MAX_DIST %f - # goals %d'%(config.DIFFICULTY, self.habitat_env.MIN_DIST, self.habitat_env.MAX_DIST, self.habitat_env._num_goals))

        self.min_measure = self.habitat_env.MAX_DIST
        self.reward_method = config.REWARD_METHOD
        if self.reward_method == 'progress':
            self.get_reward = self.get_progress_reward
        elif self.reward_method == 'milestone':
            self.get_reward = self.get_milestone_reward
        elif self.reward_method == 'coverage':
            self.get_reward = self.get_coverage_reward

        self.run_mode = 'RL'
        self.number_of_episodes = 1000
        self.need_gt_action = False
        self.has_log_info = None

    def swith_run_mode(self, mode):
        self.run_mode = mode
        self.captured_episode = self.current_episode

    def update_graph(self, node_list, affinity, changed_info, curr_info):
        if self.mapper is not None:
            self.mapper.update_graph(node_list, affinity, changed_info, curr_info)

    def draw_activated_nodes(self, activated_node_list):
        if self.mapper is not None:
            self.mapper.highlight_activated_nodes(activated_node_list)

    def build_path_follower(self):

        self.follower = ShortestPathFollower(self._env.sim, 0.8, False)
        self.curr_goal = self.current_episode.goals[self.curr_goal_idx]

    def get_best_action(self, goal=None):
        curr_goal = goal if goal is not None else self.curr_goal.position
        act = self.follower.get_next_action(curr_goal)
        return act

    def get_dist(self, goal_position):
        dist = self.habitat_env._sim.geodesic_distance(self.current_position, goal_position)
        return dist

    @property
    def curr_goal_idx(self):
        return self.habitat_env.get_metrics()['goal_index']['curr_goal_index']
    @property
    def curr_goal(self):
        return self.current_episode.goals[self.curr_goal_idx]

    def reset(self):
        #tic = time.time()
        self._previous_action = -1
        self.time_t = 0
        observations = super().reset()

        #tt = time.time()
        #self.curr_goal = self.current_episode.goals[self.curr_goal_idx]
        self.num_goals = len(self.current_episode.goals)
        self._previous_measure = self.get_dist(self.curr_goal.position)
        #print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%get metric', time.time() - tt)
        self.info = None#self.get_info(observations)
        self.total_reward = 0
        self.progress = 0
        self.stuck = 0
        self.min_measure = self.habitat_env.MAX_DIST
        self.prev_coverage = 0
        if self.need_gt_action:
            if hasattr(self.habitat_env._sim,'habitat_config'):
                sim_scene = self.habitat_env._sim.habitat_config.SCENE
            else:
                sim_scene = self.habitat_env._sim.config.SCENE
            if self.follower is None or sim_scene != self.follower._current_scene:
                self.build_path_follower()#print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%reset time', time.time()-tic)
            observations.update({'gt_action': self.get_best_action(self.current_episode.goals[0].position) - 1})
        self.positions = [self.current_position]
        self.obs = self.process_obs(observations)
        self.has_log_info = None

        if self.agent_task != 'search':
            self.log_success = [0.0 for _ in range(self.num_goals)]
            self.log_spl = [0.0 for _ in range(self.num_goals)]
            self.log_softspl = [0.0 for _ in range(self.num_goals)]
            self.curr_stage = 'search'
        return self.obs

    @property
    def scene_name(self):
        if hasattr(self.habitat_env._sim, 'habitat_config'):
            sim_scene = self.habitat_env._sim.habitat_config.SCENE
        else:
            sim_scene = self.habitat_env._sim.config.SCENE
        return sim_scene
        

    def process_obs(self, obs):
        copy_from_obs = ['target_goal', 'panoramic_rgb', 'panoramic_depth', 'rgb', 'depth']
        obs_dict = { 'step': self.time_t,
                'position': self.current_position,
                'target_pose': self.curr_goal.position,
                'distance': self.get_dist(self.curr_goal.position)}
        for key in copy_from_obs:
            if key in obs.keys():
                if key == 'target_goal':
                    obs_dict.update({key: obs[key][self.curr_goal_idx]})
                else: obs_dict.update({key: obs[key]})
        if self.need_gt_action:
            obs_dict.update(obs['gt_action'])
        if hasattr(self,'unexp'):
            obs_dict.update({'unexplored': self.unexp.astype(np.float32),
                'neighbors': self.neighbor.astype(np.float32),})
        if self.return_have_been:
            if len(self.positions) < 10:
                have_been = 0
            else:
                dists = np.linalg.norm(np.array(self.positions) - self.current_position, axis=1)
                far = np.where(dists > 1.0)[0]
                near = np.where(dists[:-10] < 1.0)[0]
                if len(far) > 0 and len(near) > 0 and (near < far.max()).any():
                    have_been = 1
                else:
                    have_been = 0
            obs_dict.update({'have_been': np.array([have_been])})
        if self.return_target_dist_score:
            target_dist_score = np.maximum(1-obs_dict['distance']/2.,0.0)
            obs_dict.update({'target_dist_score': np.array([target_dist_score])})
        return obs_dict

    def step(self, action):
        if isinstance(action, int):
            action = {'action': action}
        self._previous_action = action
        if NOISY:
            action = {'action':self.action_dict[action['action']]}
        if self.agent_task != 'search' and 'STOP' in self.action_space.spaces and action['action'] == 0:
            dist = self.get_dist(self.curr_goal.position)
            print(dist, self.success_distance)
            if dist <= self.success_distance:
                self.log_success[self.curr_goal_idx] = 1.0
                self.log_spl[self.curr_goal_idx] = self.habitat_env.task.measurements.get_metrics()['spl']
                self.log_softspl[self.curr_goal_idx] = self.habitat_env.task.measurements.get_metrics()['softspl']
                all_done = self.habitat_env.task.measurements.measures['goal_index'].increase_goal_index()
                state = self.habitat_env.sim.get_agent_state()
                obs = self.habitat_env._sim.get_observations_at(state.position, state.rotation)
                obs.update(self.habitat_env.task.sensor_suite.get_observations(
                    observations=obs,
                    episode=self.habitat_env.current_episode,
                    action=action,
                    task=self.habitat_env.task,
                ))
                if all_done:
                    done = True
                    reward = self.config.SUCCESS_REWARD
                else:
                    done = False
                    reward = 0
            else:
                obs, reward, done, self.info = super().step(action)
        else:
            obs, reward, done, self.info = super().step(action)

        self.time_t += 1
        self.info['length'] = self.time_t * done
        self.info['episode'] = int(self.current_episode.episode_id)
        self.info['distance_to_goal'] = self._previous_measure
        self.info['step'] = self.time_t
        if self.need_gt_action:
            best_action =  self.get_best_action(self.curr_goal.position)
            gt_action = best_action - 1 if best_action is not None else 0
            obs.update({'gt_action': gt_action})
        self.positions.append(self.current_position)
        self.obs = self.process_obs(obs)
        self.total_reward += reward
        if self._episode_success():
            done = True
        if self.agent_task != 'search':
            self.info.update({'success':self.log_success,
                              'spl': self.log_spl,
                              'softspl': self.log_softspl})
        return self.obs, reward, done, self.info

    def get_reward_range(self):
        return (
            self.config.SLACK_REWARD - 1.0,
            self.config.SUCCESS_REWARD + 1.0,
        )

    def get_progress_reward(self, observations):
        reward = self.config.SLACK_REWARD
        current_measure = self.get_dist(self.curr_goal.position)
        # absolute decrease on measure
        self.move = self._previous_measure - current_measure
        #print(self.move)
        if abs(self.move) < 0.01:
            self.stuck += 1
        else:
            self.stuck = 0
        self.progress = max(self.move,0.0) * 0.2
        reward += self.progress

        self._previous_measure = current_measure
        if self._episode_success():
            reward += self.config.SUCCESS_REWARD * self._env.get_metrics()['spl']
        #if self._part_success():
        #    reward += self.config.SUCCESS_REWARD
        return reward

    def get_milestone_reward(self, observations):
        reward = self.config.SLACK_REWARD
        current_measure = self.get_dist(self.curr_goal.position)
        # absolute decrease on measure
        self.move = self.min_measure - current_measure
        if abs(self.move) < 0.01:
            self.stuck += 1
        else:
            self.stuck = 0
        self.progress = max(self.move,0.0)
        reward += self.progress
        self.min_measure = min(self.min_measure, current_measure)
        if self._episode_success():
            reward += self.config.SUCCESS_REWARD * self._env.get_metrics()['spl']
        return reward


    def _episode_success(self):
        if self.num_goals == 1:
            return self._env.get_metrics()['success']
        else:
            return self.log_success[-1]


        #all_done = self.curr_goal_idx == self.num_goals - 1
        #close = self.get_dist(self.curr_goal.position) < self.success_distance
        #return all_done and close

    def _part_success(self):
        close = self.get_dist(self.curr_goal.position) < self.success_distance
        return close

    def get_success(self):
        return self._episode_success()

    def get_done(self, observations):
        done = False
        if self._env.episode_over or self._episode_success():
            done = True
        if self.stuck > 100 :
            done = True
        return done

    def get_info(self, observations):
        info = self.habitat_env.get_metrics()
        return info

    @property
    def current_position(self):
        return self.habitat_env.sim.get_agent_state().position
    def get_episode_over(self):
        return self._env.episode_over
    def get_agent_state(self):
        return self.habitat_env.sim.get_agent_state()
    def get_curr_goal_index(self):
        return self.curr_goal_idx

    def log_info(self, log_type='str', info=None):
        self.has_log_info = {'type': log_type,
                             'info': info}

    def render(self, mode='rgb'):
        info = self.get_info(None) if self.info is None else self.info
        img = observations_to_image(self.obs, info, mode='panoramic')
        str_action = 'NN'
        if 'STOP' not in self.habitat_env.task.actions:
            action_list = ["MF", 'TL', 'TR']
        else:
            action_list = ["STOP", "MF", 'TL', 'TR']
        if self._previous_action != -1:
            str_action = action_list[self._previous_action['action']]

        dist = self.get_dist(self.curr_goal.position)
        txt = 't: %03d, r: %.2f ,dist: %.2f, stuck: %d a: %s '%(self.time_t,self.total_reward, dist, self.stuck, str_action)
        if self.has_log_info is not None:
            if self.has_log_info['type'] == 'str':
                txt += ' ' + self.has_log_info['info']
        elif self.return_have_been:
            txt += '                                 '
        if hasattr(self.mapper, 'node_list'):
            if self.mapper.node_list is None:
                txt += ' node : NNNN'
                txt += ' curr : NNNN'
            else:
                num_node = len(self.mapper.node_list)
                txt += ' node : %03d' % (num_node)
                curr_info = self.mapper.curr_info
                if 'curr_node' in curr_info.keys():
                    txt += ' curr: %02d,'%(curr_info['curr_node'])
                if 'goal_prob' in curr_info.keys():
                    txt += ' goal %.3f'%(curr_info['goal_prob'])

        img = append_text_to_image(img, txt)

        if mode == 'rgb' or mode == 'rgb_array':
            return img
        elif mode == 'human':
            cv2.imshow('render', img[:,:,::-1])
            cv2.waitKey(1)
            return img
        return super().render(mode)

    def get_coverage_reward(self, observations):
        top_down_map = self.habitat_env.get_metrics()['top_down_map']
        fow = top_down_map["fog_of_war_mask"]
        self.map_size = (top_down_map['map'] != 0).sum()
        self.curr_coverage = np.sum(fow)
        new_pixel = self.curr_coverage - self.prev_coverage
        reward = np.clip(new_pixel, 0, 50) / 1000  # 0 ~ 0.1
        self.prev_coverage = self.curr_coverage

        reward += self.config.SLACK_REWARD
        current_measure = self.get_dist(self.curr_goal.position)
        # absolute decrease on measure
        self.move = self._previous_measure - current_measure
        #print(self.move)
        if abs(self.move) < 0.01:
            self.stuck += 1
        else:
            self.stuck = 0

        self._previous_measure = current_measure
        if self._episode_success():
            reward += self.config.SUCCESS_REWARD# * self._env.get_metrics()['spl']

        return reward


if __name__ == '__main__':

    from env_utils.make_env_utils import construct_envs
    from IL_configs.default import get_config
    import numpy as np
    import os
    import time
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    config = get_config('IL_configs/lgmt.yaml')
    config.defrost()
    config.DIFFICULTY = 'hard'
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_EPISODES = 10
    config.NUM_PROCESSES = 1
    config.NUM_VAL_PROCESSES = 0
    config.freeze()
    action_list = config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS
    env = construct_envs(config, eval(config.ENV_NAME), mode='single')
    obs = env.reset()
    img = env.render('rgb')
    done = False
    fps = {}
    reset_time = {}

    scene = env.env.current_episode.scene_id.split('/')[-2]
    fps[scene] = []
    reset_time[scene] = []
    imgs = [img]
    while True:
        action = env.env.get_best_action()
        #action = env.action_space.sample()
        #action = action_list.index(action['action'])
        img = env.render('rgb')
        #imgs.append(img)

        cv2.imshow('render', img[:, :, [2, 1, 0]])
        key = cv2.waitKey(0)
        #
        # if key == ord('s'): action = 1
        # elif key == ord('w'): action = 0
        # elif key == ord('a'): action = 1
        # elif key == ord('d'): action = 2
        # elif key == ord('r'):
        #     done = True
        #     print(done)
        # elif key == ord('q'):
        #     break
        # else:
        #     action = env.action_space.sample()

        if done:
            tic = time.time()
            obs = env.reset()
            toc = time.time()
            scene = env.env.current_episode.scene_id.split('/')[-2]
            fps[scene] = []
            reset_time[scene] = []
            reset_time[scene].append(toc-tic)
            done = False
            #shapes = [img.shape for img in imgs]
            #for i,im in enumerate(imgs):
            #    if im.shape != imgs[0].shape:
            #        imgs[i] = cv2.resize(im,dsize=(imgs[0].shape[1],imgs[0].shape[0]))
            #images_to_video(imgs, output_dir='.', video_name='%s_%s_no_topmap'% (scene, env.current_episode.episode_id), fps=60)
            imgs = []
            #print(env.current_episode)

        else:
            tic = time.time()
            obs, reward, done, info = env.step({'action':action})
            toc = time.time()
            fps[scene].append(toc-tic)

        #break
        if len(fps) > 20:
            break
    print('===============================')
    print('FPS : ', [(key, np.array(fps_list).mean()) for key, fps_list in fps.items()])
    print('Reset : ', [(key, np.array(reset_list).mean()) for key, reset_list in reset_time.items()])
