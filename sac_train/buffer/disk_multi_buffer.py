import os
import json
import random
import time

class Replay_Buffer(object):
    def __init__(self, buffer_size, transition, result_fold, batch_size, buffer_name):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.transition = transition
        # self.transition = namedtuple(
        #     'transition', ('global_state', 'state', 'robot_state', 'action', 'mask', 'reward', 'next_state', 'next_state_robot'))
        self.store_number = 0
        self.trainsition_fold = os.path.join(result_fold, buffer_name)
        if not os.path.exists(self.trainsition_fold):
            os.makedirs(self.trainsition_fold)


    def store_transition(self, *args):
        """Saves a transition."""
        data = self.transition(*args)
        json_data = json.dumps(data._asdict())
        save_file_name = "transition_" + str(self.store_number % self.buffer_size) + ".json"
        save_file_name = os.path.join(self.trainsition_fold, save_file_name)
        json_file = open(save_file_name, "w")
        json_file.write(json_data)
        json_file.close()
        self.store_number += 1


    def sample_data_from_disk(self):
        sample_data = []
        file_list = os.listdir(self.trainsition_fold)
        memory_size = len(file_list)
        if memory_size > 100:
            # print("*" * 100)
            selected_data_list = random.sample(range(0, memory_size), self.batch_size)
            # print("start load data from disk:", self.name, "memory_number: ", memory_size)
            for select_num in selected_data_list:
                json_file_name = file_list[select_num]
                json_file_name = os.path.join(self.trainsition_fold, json_file_name)
                file_time = os.stat(json_file_name).st_mtime
                time_now = time.time()
                time_difference = time_now - file_time
                if os.path.exists(json_file_name) and time_difference > 200:
                    f = open(json_file_name, "r")
                    content = f.read()
                    json_data = json.loads(content)
                    state_transition = self.transition(**json_data)
                    sample_data.append(state_transition)
                # else:
                #     print(json_file_name, " not saved")
            if len(sample_data) > 0:
                print("finish load data from disk:", len(sample_data))
        return sample_data