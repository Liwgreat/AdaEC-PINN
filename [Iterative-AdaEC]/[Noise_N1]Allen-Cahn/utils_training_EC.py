import copy
import random
import os, json
import numpy as np

import torch
from torch import nn, optim, autograd

from torch.nn import functional as F
import torch.nn.init as init

def set_random_seed(seed: int):
    """
    Set a fixed random seed to ensure reproducibility of experiments.
    """
    # Python built-in random generator seed
    random.seed(seed)
    
    # NumPy random generator seed
    np.random.seed(seed)
    
    # PyTorch CPU random generator seed
    torch.manual_seed(seed)
    
    # If using GPU, set the CUDA random seed for all devices
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Configure cuDNN to use deterministic algorithms for reproducibility
    torch.backends.cudnn.deterministic = True
    # Disable cuDNN benchmarking to further guarantee reproducibility
    torch.backends.cudnn.benchmark = False
    
class NetSetting:
    """
    Class for storing network configuration.
    """

    def __init__(self,
                 input_dims,
                 hidden_neurons_list,
                 output_dims,
                 hidden_activation,
                 output_activation=None,
                 initializer_method='xavier'):
        """
        Initialize a NetSetting instance.

        Args:
            input_dims (int): Number of neurons in the input layer.
            hidden_neurons_list (list): List specifying the number of neurons in each hidden layer.
            output_dims (int): Number of neurons in the output layer.
            hidden_activation (str): Name of the activation function to use in hidden layers.
            output_activation (str, optional): Name of the activation function to use in the output layer.
            initializer_method (str): Weight initialization method.
        """
        self.input_dims = input_dims
        self.hidden_neurons_list = hidden_neurons_list
        self.output_dims = output_dims
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.initializer_method = initializer_method


def sigmoid_tanh(x_input):
    return torch.sigmoid(torch.tanh(x_input))


def get_activation_function(activation_name):
    """
    Return the activation function corresponding to the given name.

    Args:
        activation_name (str): Name of the activation function.

    Returns:
        callable: The corresponding PyTorch activation function.

    Raises:
        ValueError: If the activation function name is not supported.
    """
    if activation_name == 'tanh':
        return torch.tanh
    elif activation_name == 'sin':
        return torch.sin
    elif activation_name == 'relu':
        return torch.relu
    elif activation_name == 'sigmoid(tanh)':
        return sigmoid_tanh
    else:
        raise ValueError(f"Unsupported activation function: {activation_name}")



class HiddenLayers(nn.Module):
    """
    Defines a hidden layer consisting of a single linear transformation
    followed by an activation function.
    """

    def __init__(self, net_settings, input_number, output_number):
        super(HiddenLayers, self).__init__()
        self.layer = nn.Linear(input_number, output_number)
        self.activation = get_activation_function(
            net_settings.hidden_activation)

    def forward(self, x):
        x = self.layer(x)
        x = self.activation(x)
        return x


class get_mlp_pinn(nn.Module):
    """
    Construct a multi-layer perceptron with multiple hidden layers.
    """

    def __init__(self, net_settings):
        super(get_mlp_pinn, self).__init__()
        self.net_settings = net_settings

        self.stack = nn.ModuleList()

        # First layer from input dimension to first hidden dimension
        self.stack.append(
            HiddenLayers(net_settings, net_settings.input_dims,
                         net_settings.hidden_neurons_list[0]))

        # Additional hidden layers
        for i in range(1, len(net_settings.hidden_neurons_list)):
            self.stack.append(
                HiddenLayers(net_settings,
                             net_settings.hidden_neurons_list[i - 1],
                             net_settings.hidden_neurons_list[i]))

        # Output layer
        self.stack.append(
            nn.Linear(net_settings.hidden_neurons_list[-1],
                      net_settings.output_dims))

    def forward(self, x):
        for m in self.stack:
            x = m(x)
        if self.net_settings.output_activation:
            x = get_activation_function(self.net_settings.output_activation)(x)
        return x


def initialize_weights(model, method='xavier'):
    """
    Initialize the weights of a model using the specified method.

    Args:
        model (torch.nn.Module): The model whose weights will be initialized.
        method (str): The initialization technique to use (e.g., 'xavier', 'kaiming').
    """
    if method == 'xavier':
        for name, param in model.named_parameters():
            if 'weight' in name:
                init.xavier_uniform_(param)


def compute_higher_order_derivatives(u, input_vars_list):
    """
    Compute higher-order derivatives of an output tensor with respect to a sequence of input variables.

    Args:
        u (torch.Tensor): The output tensor, typically the output of a neural network.
        input_vars_list (list of torch.Tensor): A list of input variable tensors specifying the order of differentiation.
            For example, [x, y, y] computes ∂³u / ∂x ∂y².

    Returns:
        torch.Tensor: The resulting derivative tensor after applying the specified sequence of derivatives.
    """
    grad = u
    for input_vars in input_vars_list:
        grad = autograd.grad(outputs=grad,
                             inputs=input_vars,
                             grad_outputs=torch.ones_like(grad),
                             create_graph=True,
                             retain_graph=True,
                             only_inputs=True)[0]
    return grad


def relative_l2_torch(u_pred, u_real, p_value=2):

    l2 = torch.norm(u_real - u_pred, p_value) / torch.norm(u_real, p_value)

    return l2.item()


def relative_l2_numpy(u_pred, u_real, ord_value=2):

    l2 = np.linalg.norm(u_real - u_pred, ord_value) / np.linalg.norm(
        u_real, ord_value)

    return l2


def numpy_to_tensor(data,
                    var_name,
                    value_range_dim=None,
                    to_torch=None,
                    to_cuda=None,
                    requires_grad=None):

    if value_range_dim is True:
        for col_i in range(data.shape[1]):
            min_val = np.min(data[:, col_i])
            max_val = np.max(data[:, col_i])
            print(
                f"{var_name}: Column {col_i}: range from {min_val} to {max_val}"
            )

    if to_torch is True:
        data = torch.from_numpy(data).float()

    if to_cuda is True:
        data = data.cuda()

    if requires_grad is True:
        data.requires_grad_()

    return data

def switch_optimizer(PINNs, C, Theta_list, is_training):
    """
    Enable or disable gradient computation for PINNs, C, and Theta_list based on the training mode.

    Args:
        PINNs (nn.Module): Physics-informed neural network model.
        C (torch.Tensor): Parameter tensor C.
        Theta_list (nn.Module): Collection of EC factor parameters.
        is_training (str): Training mode, 'original' to train PINNs and C, or 'EC' to train Theta_list.

    Raises:
        ValueError: If is_training is neither 'original' nor 'EC'.
    """
    if is_training == 'original':
        # Original mode: enable gradients for PINNs and C, disable for Theta_list
        for p in PINNs.parameters():
            p.requires_grad = True
        C.requires_grad = True
        Theta_list.requires_grad = False

    elif is_training == 'EC':
        # EC mode: enable gradients for Theta_list, disable for PINNs and C
        for p in PINNs.parameters():
            p.requires_grad = False
        C.requires_grad = False
        Theta_list.requires_grad = True

    else:
        raise ValueError(f"Invalid training mode: {is_training!r}. Expected 'original' or 'EC'.")

        
def train_for_error_correction(PINNs, C, optimizer_error_input, Theta_list_input, 
                               data_inputs, output_transform = None):
    
    observe_data_input = data_inputs['observe_data']
    observe_u_input = data_inputs['observe_u']
    noise_u_input = data_inputs['noise_u']
    
    switch_optimizer(PINNs, C, Theta_list_input, is_training = 'EC')
    
    #########loss error correction#########
    E_observation = PINNs(observe_data_input)
    if output_transform is not None:
        E_observation = output_transform(observe_data_input, E_observation)

    loss_error_correction = torch.mean(torch.square(E_observation + Theta_list_input - observe_u_input))
    
    #########loss noise elimination#########
    loss_noise_elimination = torch.mean(torch.square(Theta_list_input - noise_u_input))    
    
    optimizer_error_input.zero_grad()
    loss_error_correction.backward()
    optimizer_error_input.step()
    
    switch_optimizer(PINNs, C, Theta_list_input, is_training = 'original')

    return Theta_list_input, loss_noise_elimination

def flatten_line(data):
    for i in range(1, len(data)):
        if data[i] > data[i - 1]:
            data[i] = data[i - 1]
    return data

def training_recall(it_iter_input, PINNs, C, optimizer_pinn_input,
                    training_recorder_input, if_timing_point, threshold, window_wide):
    
    # --------------------------- Check conditions ---------------------------
    loss_f_for_observation_1 = training_recorder_input.loss_f_for_observation_1
    loss_f_for_observation_1_smooth = flatten_line(loss_f_for_observation_1.copy())
    
    current_min = min(loss_f_for_observation_1_smooth)
    current_min_index = loss_f_for_observation_1_smooth.index(current_min)

    # --------------------------- Record state ---------------------------
    best_PINNs_state = training_recorder_input.best_PINNs_state
    if best_PINNs_state is None or current_min < best_PINNs_state['min_loss'] + threshold:
        
        best_PINNs_state = {
            'model_state': copy.deepcopy(PINNs.state_dict()),
            'C_state': C.clone().detach(),
            'optimizer_state': copy.deepcopy(optimizer_pinn_input.state_dict()),
            'it': it_iter_input,
            'min_loss': current_min}
        
        training_recorder_input.best_PINNs_state = best_PINNs_state
    
    # --------------------------- Configure rollback ---------------------------
    if len(loss_f_for_observation_1_smooth) - current_min_index > window_wide:
        # 1. PINNs
        PINNs.load_state_dict(best_PINNs_state['model_state'])
        # 2. C
        C_value = best_PINNs_state['C_state']
        with torch.no_grad():
            C.copy_(C_value)
        # 3. optimizer_pinn
        optimizer_pinn_input.load_state_dict(best_PINNs_state['optimizer_state'])
        # 4. recall it
        timing_point = best_PINNs_state['it']
        it_iter_input = timing_point
        print('Restoring best model to the timing point:', timing_point)
        # 5. Roll back training_recorder_input to the saved iteration
        training_recorder_input.recall(timing_point)
        if_timing_point = True
        
    return (PINNs, C, it_iter_input, optimizer_pinn_input, if_timing_point)

# Define a class to store training records
class training_recorder:
    def __init__(self):
        # PDE-related losses
        self.loss_f_1 = []
        self.loss_f_for_collocation_1 = []
        self.loss_f_for_observation_1 = []
        self.loss_f_for_test_1 = []
        # EC-related losses
        self.loss_error_correction = []
        self.loss_noise_elimination = []
        # Observation losses
        self.loss_T_observation_1 = []
        self.loss_T_clear_observation_1 = []
        # Test dataset MSE
        self.loss_T_test_1 = []
        # Parameter estimates and relative test loss
        self.C1_list = []
        self.test_loss_1 = []
        # for detecting the timing point
        self.best_PINNs_state = None

    def append(self,
               pde_loss,                  # PDE residual loss
               pde_loss_collocation,      # PDE loss on collocation data
               pde_loss_observation,      # PDE loss on observational data
               pde_loss_test,             # PDE loss on testing data
               error_correction_loss,     # loss on error_correction
               noise_elimination_loss,    # loss on noise_elimination
               noisy_obs_loss,            # noisy observation MSE
               clear_obs_loss,            # clear observation MSE
               test_mse_loss,             # test data MSE
               parameter_C1,              # estimated parameter C1
               relative_test_l2           # relative L2 error on test data
               ):
        self.loss_f_1.append(pde_loss)
        self.loss_f_for_collocation_1.append(pde_loss_collocation)
        self.loss_f_for_observation_1.append(pde_loss_observation)
        self.loss_f_for_test_1.append(pde_loss_test)
        self.loss_error_correction.append(error_correction_loss)
        self.loss_noise_elimination.append(noise_elimination_loss)
        self.loss_T_observation_1.append(noisy_obs_loss)
        self.loss_T_clear_observation_1.append(clear_obs_loss)
        self.loss_T_test_1.append(test_mse_loss)
        self.C1_list.append(parameter_C1)
        self.test_loss_1.append(relative_test_l2)

    def to_dict(self) -> dict:
        """
        Convert all recorded lists into a single dictionary.
        """
        return {
            'loss_f_1': self.loss_f_1,
            'loss_f_for_collocation_1': self.loss_f_for_collocation_1,
            'loss_f_for_observation_1': self.loss_f_for_observation_1,
            'loss_f_for_test_1': self.loss_f_for_test_1,
            'loss_error_correction': self.loss_error_correction,
            'loss_noise_elimination': self.loss_noise_elimination,
            'loss_T_observation_1': self.loss_T_observation_1,
            'loss_T_clear_observation_1': self.loss_T_clear_observation_1,
            'loss_T_test_1': self.loss_T_test_1,
            'C1_list': self.C1_list,
            'test_loss_1': self.test_loss_1
        }

    def save(self, directory: str, file_name: str = 'training_records.json'):
        """
        Save all recorded data as a JSON file in the specified directory.

        :param directory: Path to the folder where the JSON file will be saved.
        :param file_name: Name of the JSON file (default: 'training_records.json').
        """
        # Create the target directory if needed
        os.makedirs(directory, exist_ok=True)
        # Gather records
        records_dict = self.to_dict()
        # Write to JSON
        full_path = os.path.join(directory, file_name)
        with open(full_path, 'w') as file_handle:
            json.dump(records_dict, file_handle, indent=4)
            
    def recall(self, recall_it: int):
        """
        Trim all recorded lists to only include data up to iteration `recall_it`.

        :param recall_it: The index (inclusive) up to which records are kept.
        """
        self.loss_f_1 = self.loss_f_1[:recall_it + 1]
        self.loss_f_for_collocation_1 = self.loss_f_for_collocation_1[:recall_it + 1]
        self.loss_f_for_observation_1 = self.loss_f_for_observation_1[:recall_it + 1]
        self.loss_f_for_test_1 = self.loss_f_for_test_1[:recall_it + 1]
        self.loss_error_correction = self.loss_error_correction[:recall_it + 1]
        self.loss_noise_elimination = self.loss_noise_elimination[:recall_it + 1]
        self.loss_T_observation_1 = self.loss_T_observation_1[:recall_it + 1]
        self.loss_T_clear_observation_1 = self.loss_T_clear_observation_1[:recall_it + 1]
        self.loss_T_test_1 = self.loss_T_test_1[:recall_it + 1]
        self.C1_list = self.C1_list[:recall_it + 1]
        self.test_loss_1 = self.test_loss_1[:recall_it + 1]


def training_calculator(PINNs, get_loss_f, Theta_list_input, recorder, data_inputs, output_transform = None):
    """
    Compute various loss metrics for a Physics-Informed Neural Network (PINN)
    and record them using the provided recorder.

    :param PINNs: PINN model to evaluate
    :param get_loss_f: calculate loss_f
    :param Theta_list_input: EC factors--Theta_list
    :param recorder: PINNsRecorder instance to store metrics
    :param data_inputs: dict containing keys:
        - 'x_inside_all', 't_inside_all'
        - 'x_inside', 't_inside'
        - 'observe_data', 'noise_u', 'observe_u', 'observe_clear_u', 
        - 'observe_data_x_inside', 'observe_data_t_inside'
        - 'test_data', 'test_u', 'test_data_x_inside', 'test_data_t_inside'
        - 'C1'
    :return: relative L2 error on the test data
    """
    # 1. PDE residual loss

    #########loss f#########
    loss_f  = get_loss_f(data_inputs['x_inside_all'],
                         data_inputs['t_inside_all'], 
                         PINNs, data_inputs['C1']).detach()
    
    #########loss f  for collocation data#########
    loss_f_collocation = get_loss_f(data_inputs['x_inside'],
                                    data_inputs['t_inside'], 
                                    PINNs, data_inputs['C1']).detach()
    
    #########loss f  for observation data#########
    loss_f_observation = get_loss_f(data_inputs['observe_data_x_inside'],
                                    data_inputs['observe_data_t_inside'],
                                    PINNs, data_inputs['C1']).detach()
    
    #########loss f  excapt observation data#########
    loss_f_test = get_loss_f(data_inputs['test_data_x_inside'],
                             data_inputs['test_data_t_inside'], 
                             PINNs, data_inputs['C1']).detach()
    
    with torch.no_grad():
        # 2. Loss_T MSE

        #########loss T noisy observation#########
        E_observation = PINNs(data_inputs['observe_data'])
        if output_transform is not None:
            E_observation = output_transform(data_inputs['observe_data'], E_observation)
        loss_T_observation_noisy = torch.mean(torch.square(E_observation - data_inputs['observe_u']))

        #########loss error correction#########
        E_observation = PINNs(data_inputs['observe_data'])
        if output_transform is not None:
            E_observation = output_transform(data_inputs['observe_data'], E_observation)
        loss_error_correction = torch.mean(torch.square(E_observation + Theta_list_input - \
                                                        data_inputs['observe_u']))
        
        #########loss noise elimination#########
        loss_noise_elimination = torch.mean(torch.square(Theta_list_input - data_inputs['noise_u']))    
        
        #########loss T clear observation#########        
        E_observation_clear = PINNs(data_inputs['observe_data'])
        if output_transform is not None:
            E_observation_clear = output_transform(data_inputs['observe_data'],E_observation_clear) 
        loss_T_observation_clear = torch.mean(torch.square(E_observation_clear-data_inputs['observe_clear_u']))
        
        #########loss T excapt observation#########
        E_observation_excapt = PINNs(data_inputs['test_data'])
        if output_transform is not None:
            E_observation_excapt = output_transform(data_inputs['test_data'], E_observation_excapt)
        loss_T_test = torch.mean(torch.square(E_observation_excapt - data_inputs['test_u']))  
        
        # 3. test_loss NRMSE
        pre_u = PINNs(data_inputs['test_data'])
        if output_transform is not None:
            pre_u = output_transform(data_inputs['test_data'], pre_u)
        test_loss = relative_l2_torch(pre_u, data_inputs['test_u'])

    # 5. Record all metrics
    recorder.append(
        pde_loss = loss_f.item(),  # PDE residual loss
        pde_loss_collocation = loss_f_collocation.item(),  # PDE loss on collocation data
        pde_loss_observation = loss_f_observation.item(),  # PDE loss on observational data
        pde_loss_test = loss_f_test.item(),  # PDE loss on testing data
        error_correction_loss = loss_error_correction.item(),     # loss on error_correction
        noise_elimination_loss = loss_noise_elimination.item(),    # loss on noise_elimination
        noisy_obs_loss = loss_T_observation_noisy.item(),  # noisy observation MSE
        clear_obs_loss = loss_T_observation_clear.item(),  # clear observation MSE
        test_mse_loss = loss_T_test.item(),  # test data MSE
        parameter_C1 = data_inputs['C1'].item(),  # estimated parameter C1
        relative_test_l2 = test_loss  # relative L2 error on test data
    )

    return test_loss

def change_C_learning_rate(optimizer_pinn_input, C1_id_input, learning_rate = None):
    
    for param_group in optimizer_pinn_input.param_groups:

        if any(id(param) == C1_id_input for param in param_group['params']):
            
            if learning_rate is None:
                C1_lr = param_group['lr']
                print(f'Current C1 learning rate: {C1_lr}')
            else: 
                param_group['lr'] = learning_rate
                print(f'Current C1 learning rate: {learning_rate}')
                
def change_EC_factors_learning_rate(optimizer_error_input, learning_rate = None):
    
    for param_group in optimizer_error_input.param_groups:

        if learning_rate is None:
            EC_factors_lr = param_group['lr']
            print(f'Current EC_factors learning rate: {EC_factors_lr}')
        else: 
            param_group['lr'] = learning_rate
            print(f'Current EC_factors learning rate: {learning_rate}')
            
            
            
class AdaEC_class:
    
    def __init__(self, optimizer_pinn_input, optimizer_error_input,
                 min_lr, max_lr, adjustment_factor,
                 patience_base, patience_down, patience_up,
                 error_threshold):
        
        # Scenario-I 
        self.factor_down = 1/adjustment_factor
        self.patience_down = patience_down

        # Scenario-II 
        self.factor_up = adjustment_factor
        self.patience_up = patience_up    
        
        # Scenario-III 
        self.factor_C = 1/(adjustment_factor**2)
        self.patience_base = patience_base    
        
        # optimizers
        self.optimizer_pinn = optimizer_pinn_input
        self.optimizer_error = optimizer_error_input
        
        # other parameters
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.error_threshold = error_threshold
        self.eps = 1e-12 # avoid the influence of floating-point numbers
        
        # Internal state
        self.num_bad_iters_base = 0
        self.num_bad_iters_up = 0
        self.num_bad_iters_down = 0
        self.last_adjust_iter = 0
        
        # Internal metric
        self.pre_metric = None
        self.pre_base_loss = None   
            
    def step(self, training_recorder_input, it_iter_input, C1_id_input):
        
        loss_f_for_collocation_1 = training_recorder_input.loss_f_for_collocation_1
        loss_f_for_collocation_1 = np.array(loss_f_for_collocation_1[self.last_adjust_iter:]) 
        
        loss_f_for_observation_1 = training_recorder_input.loss_f_for_observation_1
        loss_f_for_observation_1 = np.array(loss_f_for_observation_1[self.last_adjust_iter:]) 
        
        now_metric = (loss_f_for_observation_1)[-1] - flatten_line(loss_f_for_collocation_1)[-1]
        now_loss = flatten_line(loss_f_for_observation_1)[-1]

        if (self.pre_metric is not None) and (self.pre_base_loss is not None):
            
            # Scenario-III is monitored
            if now_loss == self.pre_base_loss:
                self.num_bad_iters_base += 1
                # Scenario-III is continuously monitored
                if self.num_bad_iters_base > self.patience_base:
                    print('____________________New training stagnation____________________')
                    self._adjust_C_lr("down", self.factor_C, C1_id_input)
                    self._adjust_EC_factors_lr("max_lr")
                    self.reset(state = 'new_iter', new_iter = it_iter_input)
                    
            # Scenario-I is monitored
            if now_metric < self.pre_metric - self.error_threshold:
                self.num_bad_iters_down += 1
                # Scenario-I is continuously monitored
                if self.num_bad_iters_down > self.patience_down:
                    self._adjust_EC_factors_lr("down", self.factor_down)
                    self.reset()
                    
            # Scenario-II is monitored
            elif now_metric >= self.pre_metric + self.error_threshold:
                self.num_bad_iters_up += 1
                # Scenario-II is continuously monitored
                if self.num_bad_iters_up > self.patience_up:
                    self._adjust_EC_factors_lr("up", self.factor_up)
                    self.reset()
        
        self.pre_metric = now_metric
        self.pre_base_loss = now_loss
        
    def reset(self, state = 'default', new_iter = None):
        """
        Reset scheduler to initial state.
        """
        self.num_bad_iters_base = 0
        self.num_bad_iters_up = 0
        self.num_bad_iters_down = 0
        
        if state == 'new_iter':
            self.last_adjust_iter = new_iter
        
        
    def _adjust_EC_factors_lr(self, direction, factor = None):
        
        for param_group in self.optimizer_error.param_groups:
            old_lr = param_group['lr']
            
            # Scenario-III is continuously monitored
            if direction == "max_lr":
                param_group['lr'] = self.max_lr
            else:
                #---------------------------AdaEC---------------------------
                if old_lr == 0:
                    old_lr = self.min_lr

                if direction == "down":
                    new_lr = old_lr * factor
                    if new_lr < self.min_lr:
                        new_lr = 0
                elif direction == "up":
                    new_lr = min(old_lr * factor, self.max_lr)
                #---------------------------AdaEC---------------------------
                param_group['lr'] = new_lr
       
    def _adjust_C_lr(self, direction, factor, C1_id_input):
        
        for param_group in self.optimizer_pinn.param_groups:
            # Scenario-III is continuously monitored
            if any(id(param) == C1_id_input for param in param_group['params']):
                old_lr = param_group['lr']
                #---------------------------AdaEC---------------------------
                if old_lr == 0:
                    old_lr = self.min_lr

                if direction == "down":
                    new_lr = old_lr * factor
    
                    if new_lr<=self.min_lr:
                        new_lr = 0
                #---------------------------AdaEC---------------------------
                param_group['lr'] = new_lr     


