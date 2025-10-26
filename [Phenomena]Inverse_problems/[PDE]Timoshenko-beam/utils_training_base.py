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




# Define a class to store training records
class training_recorder:
    def __init__(self):
        # PDE-related losses
        self.loss_f_1 = []
        self.loss_f_for_collocation_1 = []
        self.loss_f_for_observation_1 = []
        self.loss_f_for_test_1 = []
        # Observation losses
        self.loss_T_observation_1 = []
        self.loss_T_clear_observation_1 = []
        # Test dataset MSE
        self.loss_T_test_1 = []
        # Parameter estimates and relative test loss
        self.C1_list = []
        self.test_loss_1 = []

    def append(self,
               pde_loss,                  # PDE residual loss
               pde_loss_collocation,      # PDE loss on collocation data
               pde_loss_observation,      # PDE loss on observational data
               pde_loss_test,             # PDE loss on testing data
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
            'loss_T_observation_1': self.loss_T_observation_1,
            'loss_T_clear_observation_1': self.loss_T_clear_observation_1,
            'loss_T_test_1': self.loss_T_test_1,
            'C1_list': self.C1_list,
            'test_loss_1': self.test_loss_1
        }

    def save(self, directory: str, file_name: str = 'training_records.json', 
                                   EC_method = False, records_dict_input = None):
        """
        Save all recorded data as a JSON file in the specified directory.

        :param directory: Path to the folder where the JSON file will be saved.
        :param file_name: Name of the JSON file (default: 'training_records.json').
        """
        # Create the target directory if needed
        os.makedirs(directory, exist_ok=True)
        # Gather records
        if EC_method == False:
            records_dict = self.to_dict()
        else:
            records_dict = records_dict_input.copy()
        # Write to JSON
        full_path = os.path.join(directory, file_name)
        with open(full_path, 'w') as file_handle:
            json.dump(records_dict, file_handle, indent=4)
            

def training_calculator(PINNs, get_loss_f, recorder, data_inputs, output_transform = None):
    """
    Compute various loss metrics for a Physics-Informed Neural Network (PINN)
    and record them using the provided recorder.

    :param PINNs: PINN model to evaluate
    :param get_loss_f: calculate loss_f
    :param recorder: PINNsRecorder instance to store metrics
    :param data_inputs: dict containing keys:
        - 'x_inside_all', 't_inside_all'
        - 'x_inside', 't_inside'
        - 'observe_data', 'observe_u', 'observe_clear_u', 'observe_data_x_inside', 'observe_data_t_inside'
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
        noisy_obs_loss = loss_T_observation_noisy.item(),  # noisy observation MSE
        clear_obs_loss = loss_T_observation_clear.item(),  # clear observation MSE
        test_mse_loss = loss_T_test.item(),  # test data MSE
        parameter_C1 = data_inputs['C1'].item(),  # estimated parameter C1
        relative_test_l2 = test_loss  # relative L2 error on test data
    )

    return test_loss