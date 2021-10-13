import copy
import time
import numpy as np
import torch as th


def launch(args, container_device, training_start_time, t_env, _learner,
           learner_queue, learner_queue_size, priority_update_queue,
           container_id, container_qs, centralizer_mac, mac, target_mac, mixer, target_mixer,
           centralizer_mac_time_step):
    learner = copy.deepcopy(_learner)
    if container_device != "cpu":
        th.cuda.set_device(container_device)
        learner.cuda()

    last_container_qs_update_time = time.time()

    ################### log stuff ####################
    if args.log_container_learner:
        log_last_time = time.time()
        log_frequency = 60
        log_n = 200
        log_i = 0
        log_wait_times = []
        log_wait_time = None
    ##################################################

    # TODO: add a self buffer stores batches, needed when learner queue is empty
    while time.time() - training_start_time < args.training_time:
        if not learner_queue.empty():
            ################### log stuff ####################
            if args.log_container_learner:
                if log_wait_time is not None:
                    log_wait_time += time.time()
                    if len(log_wait_times) < log_n:
                        log_wait_times.append(log_wait_time)
                    else:
                        log_wait_times[log_i] = log_wait_time
                        log_i = (log_i + 1) % log_n

                    if time.time() - log_last_time > log_frequency:
                        log_last_time = time.time()
                        print("container id={} learner: average wainging time {}, number of model updates {}, average (loss {}, pi_kl {}, priority {}), average training time {}, average batch length {};".format(container_id, np.mean(log_wait_times), learner.cur_episode, np.mean(learner.log_loss), np.mean(learner.log_pi_kl), np.mean(learner.log_priority), np.mean(learner.log_training_time), np.mean(learner.log_avg_len)))
            ##################################################

            _ep_ids, _batch, _sample_time = learner_queue.get()
            ep_ids = copy.deepcopy(_ep_ids)
            batch = copy.deepcopy(_batch)
            sample_time = copy.deepcopy(_sample_time)
            del _ep_ids, _batch, _sample_time
            learner_queue_size -= 1

            if batch.device != container_device:
                batch.to(container_device)

            priority = learner.train(batch, int(t_env), centralizer_mac, container_qs, centralizer_mac_time_step)
            priority_update_queue.put(copy.deepcopy((ep_ids, priority, sample_time)))

            if time.time() - last_container_qs_update_time > args.update_container_qs_time_interval:
                last_container_qs_update_time = time.time()
                container_qs.load(container_id, learner.mac.agent.fc2)
            mac.load_state(learner.mac)
            target_mac.load_state(learner.target_mac)
            mixer.load_state_dict(learner.mixer.state_dict())
            target_mixer.load_state_dict(learner.target_mixer.state_dict())

            ################### log stuff ####################
            if args.log_container_learner:
                log_wait_time = -time.time()
            ##################################################
