import torch
import torch.nn as nn


########################################################################################################
## Implementation taken from with slight edits
## https://github.com/xiongjiechen/Normalizing-Flows-DPFs/blob/main/resamplers/resamplers.py
########################################################################################################

class OTResampler(nn.Module):
    # def __init__(self, epsilon=0.1, scaling=0.75, threshold=1e-3, max_iter=100):
    # def __init__(self, epsilon=0.01, scaling=0.75, threshold=1e-3, max_iter=500):
    def __init__(self, epsilon=0.01, scaling=0.9, threshold=1e-3, max_iter=500): # From OG paper code
    # def __init__(self, epsilon=0.05, scaling=0.9, threshold=1e-3, max_iter=500):  # Tried and seems to work

        super().__init__()
            
        self.epsilon = epsilon
        self.scaling = scaling
        self.threshold = threshold
        self.max_iter = max_iter

        self.resampling=resampler_ot

    def forward(self, particles, particle_probs):
        particles_resampled, particle_probs_resampled, index_p = self.resampling(particles, particle_probs, eps=self.epsilon, scaling=self.scaling, threshold=self.threshold, max_iter=self.max_iter)
        return particles_resampled, particle_probs_resampled, index_p



def print_and_exit_on_nan(x, x_name):
    if(torch.sum(torch.isnan(x)) > 0):
        print("")
        print("")
        print("{} NAN".format(x_name))
        print("")
        print("")
        exit(0)


    if(torch.sum(torch.isinf(x)) > 0):
        print("")
        print("")
        print("{} INF".format(x_name))
        print("")
        print("")
        exit(0)


def resampler_ot(particles, weights, eps=0.1, scaling=0.75, threshold=1e-3, max_iter=100, flag=torch.tensor(True,requires_grad=False)):

    device = particles.device

    logw=weights.log()
    batch_size, num_particles, dimensions = particles.shape
    particles_resampled, particle_probs_resampled, particles_probs_log_resampled=OT_resampling(particles, logw=logw, eps=eps,scaling=scaling, threshold=threshold, max_iter=max_iter, n=particles.shape[1],flag=flag)
    index_p = (torch.arange(num_particles)+num_particles* torch.arange(batch_size)[:, None].repeat((1, num_particles))).type(torch.int64).to(device)
    return particles_resampled, particle_probs_resampled, index_p

def diameter(x, y):
    diameter_x = x.std(dim=1, unbiased=False).max(dim=-1)[0]
    diameter_y = y.std(dim=1, unbiased=False).max(dim=-1)[0]
    res = torch.maximum(diameter_x, diameter_y)
    return torch.where(res == 0., 1., res.double())


def cost(x, y):
    return squared_distances(x, y) / 2.


def squared_distances(x, y):
    # tmp = torch.cdist(x, y, p=2.0, compute_mode="use_mm_for_euclid_dist") ** 2

    x1 = torch.tile(x.unsqueeze(2), [1,1,y.shape[1],1])
    y1 = torch.tile(y.unsqueeze(1), [1,x.shape[1],1,1])
    tmp2 = torch.sum((x1 - y1)**2, dim=-1)



    print_and_exit_on_nan(x, "x")
    print_and_exit_on_nan(y, "y")
    print_and_exit_on_nan(tmp2, "tmp2")


    return tmp2 


def max_min(x, y):
    max_max = torch.maximum(x.max(dim=1)[0].max(dim=1)[0], y.max(dim=1)[0].max(dim=1)[0])
    min_min = torch.minimum(x.max(dim=1)[0].min(dim=1)[0], y.min(dim=1)[0].min(dim=1)[0])

    return max_max - min_min


def softmin(epsilon, cost_matrix, f):
    """Implementation of softmin function
    :param epsilon: float regularisation parameter
    :param cost_matrix:
    :param f:
    :return:
    """
    n = cost_matrix.shape[1]
    b = cost_matrix.shape[0]

    f_ = f.reshape([b, 1, n])
    temp_val = f_ - cost_matrix / epsilon.reshape([-1, 1, 1])
    log_sum_exp = torch.logsumexp(temp_val, dim=2)
    res = -epsilon.reshape([-1, 1]) * log_sum_exp


    print_and_exit_on_nan(epsilon, "epsilon")
    print_and_exit_on_nan(f, "f")
    print_and_exit_on_nan(temp_val, "temp_val")
    print_and_exit_on_nan(log_sum_exp, "log_sum_exp")
    print_and_exit_on_nan(res, "res")



    return res


def sinkhorn_loop(log_alpha, log_beta, cost_xy, cost_yx, cost_xx, cost_yy, epsilon, particles_diameter, scaling, threshold, max_iter):

    device = log_alpha.device

    batch_size = log_alpha.shape[0]
    continue_flag = torch.ones([batch_size], dtype=bool).to(device)
    epsilon_0 = particles_diameter ** 2
    scaling_factor = scaling ** 2



    # Ali Added Hack
    epsilon_0 = torch.clamp(epsilon_0, min=1e-8)

    print_and_exit_on_nan(epsilon_0, "epsilon_0")
    print_and_exit_on_nan(cost_yx, "cost_yx")
    print_and_exit_on_nan(log_alpha, "log_alpha")


    a_y_init = softmin(epsilon_0, cost_yx, log_alpha)
    b_x_init = softmin(epsilon_0, cost_xy, log_beta)

    a_x_init = softmin(epsilon_0, cost_xx, log_alpha)
    b_y_init = softmin(epsilon_0, cost_yy, log_beta)

    print_and_exit_on_nan(a_y_init, "a_y_init")
    print_and_exit_on_nan(b_x_init, "b_x_init")
    print_and_exit_on_nan(a_x_init, "a_x_init")
    print_and_exit_on_nan(b_y_init, "b_y_init")


    def stop_condition(i, _a_y, _b_x, _a_x, _b_y, continue_, _running_epsilon):
        n_iter_cond = i < max_iter - 1
        return torch.logical_and(torch.tensor(n_iter_cond, dtype=bool).to(device), torch.all(continue_.bool()))

    def apply_one(a_y, b_x, a_x, b_y, continue_, running_epsilon):
        running_epsilon_ = running_epsilon.reshape([-1, 1])
        continue_reshaped = continue_.reshape([-1, 1])
        # TODO: Hopefully one day tensorflow controlflow will be lazy and not strict...
        at_y = torch.where(continue_reshaped, softmin(running_epsilon, cost_yx, log_alpha + b_x / running_epsilon_),a_y)
        bt_x = torch.where(continue_reshaped, softmin(running_epsilon, cost_xy, log_beta + a_y / running_epsilon_), b_x)

        at_x = torch.where(continue_reshaped, softmin(running_epsilon, cost_xx, log_alpha + a_x / running_epsilon_),a_x)
        bt_y = torch.where(continue_reshaped, softmin(running_epsilon, cost_yy, log_beta + b_y / running_epsilon_), b_y)

        a_y_new = (a_y + at_y) / 2
        b_x_new = (b_x + bt_x) / 2

        a_x_new = (a_x + at_x) / 2
        b_y_new = (b_y + bt_y) / 2

        a_y_diff = (torch.abs(a_y_new - a_y)).max(dim=1)[0]
        b_x_diff = (torch.abs(b_x_new - b_x)).max(dim=1)[0]

        local_continue = torch.logical_or(a_y_diff > threshold, b_x_diff > threshold)
        return a_y_new, b_x_new, a_x_new, b_y_new, local_continue

    def body(i, a_y, b_x, a_x, b_y, continue_, running_epsilon):
        new_a_y, new_b_x, new_a_x, new_b_y, local_continue = apply_one(a_y, b_x, a_x, b_y, continue_,
                                                                       running_epsilon)
        new_epsilon = torch.maximum(running_epsilon * scaling_factor, epsilon)
        global_continue = torch.logical_or(new_epsilon < running_epsilon, local_continue)

        return i + 1, new_a_y, new_b_x, new_a_x, new_b_y, global_continue, new_epsilon

    total_iter = 0
    converged_a_y, converged_b_x, converged_a_x, converged_b_y = a_y_init, b_x_init, a_x_init, b_y_init
    final_epsilon = epsilon_0

    while stop_condition(total_iter, converged_a_y, converged_b_x, converged_a_x, converged_b_y, continue_flag,final_epsilon):
        total_iter, converged_a_y, converged_b_x, converged_a_x, converged_b_y, continue_flag, final_epsilon = body( total_iter, converged_a_y, converged_b_x, converged_a_x, converged_b_y, continue_flag, final_epsilon)

    converged_a_y, converged_b_x, converged_a_x, converged_b_y, = converged_a_y.detach().clone(), converged_b_x.detach().clone(), converged_a_x.detach().clone(), converged_b_y.detach().clone()
    epsilon_ = epsilon.reshape([-1, 1])

    final_a_y = softmin(epsilon, cost_yx, log_alpha + converged_b_x / epsilon_)
    final_b_x = softmin(epsilon, cost_xy, log_beta + converged_a_y / epsilon_)
    final_a_x = softmin(epsilon, cost_xx, log_alpha + converged_a_x / epsilon_)
    final_b_y = softmin(epsilon, cost_yy, log_beta + converged_b_y / epsilon_)
    return final_a_y, final_b_x, final_a_x, final_b_y, total_iter + 2





def sinkhorn_potentials(log_alpha, x, log_beta, y, epsilon, scaling, threshold, max_iter):

    print_and_exit_on_nan(x, "x 1")
    print_and_exit_on_nan(y, "y 1")


    cost_xy = cost(x, y.detach().clone())
    cost_yx = cost(y, x.detach().clone())
    cost_xx = cost(x, x.detach().clone())
    cost_yy = cost(y, y.detach().clone())
    scale = max_min(x, y).detach().clone()

    print_and_exit_on_nan(cost_yx, "cost_yx 1")


    a_y, b_x, a_x, b_y, total_iter = sinkhorn_loop(log_alpha, log_beta, cost_xy, cost_yx, cost_xx, cost_yy, epsilon, scale, scaling, threshold, max_iter)

    print_and_exit_on_nan(cost_xy, "cost_xy")
    print_and_exit_on_nan(cost_yx, "cost_yx")
    print_and_exit_on_nan(cost_xx, "cost_xx")
    print_and_exit_on_nan(cost_yy, "cost_yy")
    print_and_exit_on_nan(scale, "scale")

    print_and_exit_on_nan(a_y, "a_y")
    print_and_exit_on_nan(b_x, "b_x")
    print_and_exit_on_nan(a_x, "a_x")
    print_and_exit_on_nan(b_y, "b_y")



    return a_y, b_x, a_x, b_y, total_iter


def transport_from_potentials(x, f, g, eps, logw, n):
    device = x.device
    float_n = n
    log_n = torch.log(float_n).to(device)

    cost_matrix = cost(x, x)

    fg = torch.unsqueeze(f, 2) + torch.unsqueeze(g, 1)  # fg = f + g.T
    temp = fg - cost_matrix
    temp = temp / eps

    temp = temp - torch.logsumexp(temp, dim=1, keepdims=True) + log_n

    # We "divide" the transport matrix by its col-wise sum to make sure that weights normalise to logw.
    temp = temp + torch.unsqueeze(logw, 1)
    transport_matrix = torch.exp(temp)

    return transport_matrix

def transport_function(x, logw, eps, scaling, threshold, max_iter, n):

    device = logw.device

    eps = torch.tensor(eps, dtype=torch.float).to(device)
    float_n = torch.tensor(n, dtype=torch.float).to(device)
    log_n = torch.log(float_n).to(device)
    uniform_log_weight = -log_n * torch.ones_like(logw).to(device)
    dimension = torch.tensor(x.shape[-1]).to(device)




    centered_x = x - x.mean(dim=1, keepdim=True).detach().clone()
    diameter_value = diameter(x, x)

    scale = diameter_value.reshape([-1, 1, 1]) * torch.sqrt(dimension)
    scale = torch.clamp(scale, min=1e-8)



    if(torch.sum(torch.isinf(x.mean(dim=1, keepdim=True).detach().clone())) > 0):
        print("")
        print("")

        mean = x.mean(dim=1, keepdim=True).detach().clone()

        print(torch.sum(torch.isnan(x)))
        print(torch.sum(torch.isnan(mean)))


        print(torch.sum(torch.isinf(x)))
        print(torch.sum(torch.isinf(mean)))

        print(torch.isinf(mean).shape)

        mask = torch.any(torch.isinf(mean), dim=-1)

        for i in range(mask.shape[0]):
            if(mask[i] == True):
                print(mean[i])
                print(x[i])



        # print(x.shape)
        # print(mean.shape)

        print("")
        print("")
        exit(0)



    print_and_exit_on_nan(x, "x")
    print_and_exit_on_nan(x.mean(dim=1, keepdim=True).detach().clone(), "x.mean(dim=1, keepdim=True).detach().clone()")
    print_and_exit_on_nan(centered_x, "centered_x")


    scaled_x = centered_x / scale.detach().clone()
    alpha, beta, _, _, _ = sinkhorn_potentials(logw, scaled_x, uniform_log_weight, scaled_x, eps, scaling, threshold, max_iter)
    transport_matrix = transport_from_potentials(scaled_x, alpha, beta, eps, logw, float_n)

    if(torch.sum(torch.isnan(scaled_x)) > 0):
        print("")
        print("")
        print("scaled_x NAN")
        exit(0)


    if(torch.sum(torch.isnan(alpha)) > 0):
        print("")
        print("")
        print("alpha NAN")
        exit(0)


    if(torch.sum(torch.isnan(beta)) > 0):
        print("")
        print("")
        print("beta NAN")
        exit(0)



    if(torch.sum(torch.isnan(transport_matrix)) > 0):
        print("")
        print("")
        print("transport_matrix NAN")
        exit(0)



    return transport_matrix

def transport_grad(x_original, logw, eps, scaling, threshold, max_iter, n, grad_output=None):
    device = logw.device
    transport_matrix=transport_function(x_original, logw, eps, scaling, threshold, max_iter, n).requires_grad_()
    transport_matrix.backward(grad_output)
    return x_original.grad, logw.grad

class transport(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, logw, x_, logw_, transport_matrix_):
        ctx.save_for_backward(transport_matrix_, x_, logw_)
        return transport_matrix_.clone()  # grad

    @staticmethod
    def backward(ctx, d_transport):
        d_transport=torch.clamp(d_transport, -1., 1.)
        transport_matrix_, x_, logw_ = ctx.saved_tensors
        dx, dlogw = torch.autograd.grad(transport_matrix_, [x_, logw_], grad_outputs=d_transport, retain_graph=True)
        return None, None, None, None, None


def resample(tensor, new_tensor, flags):
    ndim = len(tensor.shape)
    shape = [-1] + [1] * (ndim - 1)
    ret = torch.where(torch.reshape(flags, shape), new_tensor.float(), tensor.float())
    return ret


def apply_transport_matrix(particles, weights, log_weights, transport_matrix, flags):
    float_n_particles = torch.tensor(particles.shape[1]).float()
    transported_particles = torch.matmul(transport_matrix.float(), particles.float())
    uniform_log_weights = -float_n_particles.log() * torch.ones_like(log_weights)
    uniform_weights = torch.ones_like(weights) / float_n_particles

    resampled_particles = resample(particles, transported_particles, flags)
    resampled_weights = resample(weights, uniform_weights, flags)
    resampled_log_weights = resample(log_weights, uniform_log_weights, flags)

    return resampled_particles, resampled_weights, resampled_log_weights


def OT_resampling(x, logw, eps, scaling, threshold, max_iter, n,flag=torch.tensor(True, requires_grad=False)):

    device = logw.device

    flag = flag.to(device)
    x_, logw_ = x.detach().clone().requires_grad_(), logw.detach().clone().requires_grad_()
    transport_matrix_ = transport_function(x_, logw_, eps, scaling, threshold, max_iter, n)


    if(torch.sum(torch.isnan(x)) > 0):
        print("")
        print("")
        print("x NAN")
        exit(0)


    if(torch.sum(torch.isnan(logw)) > 0):
        print("")
        print("")
        print("logw NAN")
        exit(0)



    if(torch.sum(torch.isnan(transport_matrix_)) > 0):
        print("")
        print("")
        print("transport_matrix_ NAN")
        exit(0)




    calculate_transport=transport.apply
    transport_matrix = calculate_transport(x, logw, x_, logw_, transport_matrix_)
    resampled_particles, resampled_weights, resampled_log_weights = apply_transport_matrix(x, logw.exp(), logw, transport_matrix, flag)
    return resampled_particles, resampled_weights, resampled_log_weights