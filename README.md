# MQP-IoT-Rule-Conflict-Detector
Graph Neural Network model that takes in a list of IoT Rules and outputs the ones that conflict

The five main files are config.py, synthetic_data_maker.py, train_gnn_conflict_model.py, rule_checker.py, and rule_checker_with_verification.py. 

Config.py contains the dictionary the model was trianed on. This is the specific wording of different devices and their corresponding states and actions. It is vital that rules entered for evaluation use these devices and their states and actions.

synthetic_data_maker.py is the tool that was used to make the dataset that the model was trained on. It creates a graph (rule_graph.pt) of different IoT rules where each node represents a rule and each edge represents a relationship (chain, conflict, or none). 

train_gnn_conflict_model.py is the script that was used to build and train the model. During trianing the current best is stored in best_rule_model.pt and at the end the best model is saved as ggn_rule_model.pt. 

rule_checker.py is used to load the model to test a set of rules. Rules to be tested are entered into rules.csv and must use the same language in config.py and be in the format of trigger_device, trigger_state, action_device, action_action. Rules that chain or conflict are output into the terminal as well as results.csv along with a confidence interval. This represents the pure model output. 

rule_checker_with_verification.py is the same as rule_checker.py, but includes an extra verification step before outputting.  
