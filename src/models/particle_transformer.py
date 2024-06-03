import torch


class IdentityTransformer():
    def __init__(self):
        pass

    def forward_tranform(self, particles):
        return particles

    def backward_tranform(self, particles):
        return particles

    def apply_norm(self, particles):
        return particles


    def downscale(self, particles):
        return particles

    def upscale(self, particles):
        return particles




    def downscale_map(self, x, y):
        return x, y

    def downscale_map(self, x, y):
        return x, y


class DeepMindTransformer(IdentityTransformer):
    def __init__(self):
        pass

    def forward_tranform(self, particles):

        assert(particles.shape[-1] == 4)

        new_p = []
        new_p.append(particles[...,0])
        new_p.append(particles[...,1])
        new_p.append(torch.atan2(particles[...,2], particles[...,3]))
        new_p = torch.stack(new_p, dim=-1)

        return new_p

    def backward_tranform(self, particles):
        
        assert(particles.shape[-1] == 3)

        new_p = []
        new_p.append(particles[...,0])
        new_p.append(particles[...,1])
        new_p.append(torch.sin(particles[...,2]))
        new_p.append(torch.cos(particles[...,2]))
        new_p = torch.stack(new_p, dim=-1)

        return new_p


    def apply_norm(self, particles):

        if(particles.shape[-1] == 4):

            # Norm the last 2 dims to be 1
            mag = torch.sqrt((particles[...,2] ** 2) + (particles[...,3] ** 2))

            # Add a small eps to prevent div by 0
            mag = mag + 1e-8

            new_p = []
            new_p.append(particles[...,0])
            new_p.append(particles[...,1])
            new_p.append(particles[...,2] / mag)
            new_p.append(particles[...,3] / mag)
            new_p = torch.stack(new_p, dim=-1)
            return new_p

        else:
            return particles


    # def downscale(self, p):
    #     assert(p.shape[-1] == 3)

    #     new_p = []
    #     new_p.append(p[...,0] / 10.0)
    #     new_p.append(p[...,1] / 10.0)
    #     new_p.append(p[...,2])
    #     new_p = torch.stack(new_p, dim=-1)

    #     return new_p

    # def upscale(self, p):
    #     assert(p.shape[-1] == 3)

    #     new_p = []
    #     new_p.append(p[...,0] * 10.0)
    #     new_p.append(p[...,1] * 10.0)
    #     new_p.append(p[...,2])
    #     new_p = torch.stack(new_p, dim=-1)

    #     return new_p





class BearingsOnlyTransformer(IdentityTransformer):
    def __init__(self):
        pass

    def forward_tranform(self, particles):
        
        assert(particles.shape[-1] == 4)

        new_p = []
        new_p.append(particles[...,0])
        new_p.append(particles[...,1])
        new_p.append(torch.atan2(particles[...,2], particles[...,3]))
        new_p = torch.stack(new_p, dim=-1)

        return new_p

    def backward_tranform(self, particles):
        
        assert(particles.shape[-1] == 3)

        new_p = []
        new_p.append(particles[...,0])
        new_p.append(particles[...,1])
        new_p.append(torch.sin(particles[...,2]))
        new_p.append(torch.cos(particles[...,2]))
        new_p = torch.stack(new_p, dim=-1)

        return new_p


    def apply_norm(self, particles):

        if(particles.shape[-1] == 4):

            # Norm the last 2 dims to be 1
            mag = torch.sqrt((particles[...,2] ** 2) + (particles[...,3] ** 2))


            # Clamp
            x = torch.clamp(particles[...,0], min=-10000, max=10000)
            y = torch.clamp(particles[...,1], min=-10000, max=10000)

            new_p = []
            new_p.append(x)
            new_p.append(y)
            new_p.append(particles[...,2] / mag)
            new_p.append(particles[...,3] / mag)
            new_p = torch.stack(new_p, dim=-1)
            return new_p

        else:
            # return particles

            # Clamp
            x = torch.clamp(particles[...,0], min=-10000, max=10000)
            y = torch.clamp(particles[...,1], min=-10000, max=10000)

            new_p = []
            new_p.append(x)
            new_p.append(y)
            new_p.append(particles[...,2])
            new_p = torch.stack(new_p, dim=-1)
            return new_p



class BearingsOnlyVelocityTransformer(IdentityTransformer):
    def __init__(self):
        pass

class BearingsOnlyVectorAngleTransformer(IdentityTransformer):
    def __init__(self):
        pass

    def apply_norm(self, particles):

        # Norm the last 2 dims to be 1
        mag = torch.sqrt((particles[...,2] ** 2) + (particles[...,3] ** 2))
        # particles[...,2] = particles[...,2] / mag
        # particles[...,3] = particles[...,3] / mag



        new_p = []
        new_p.append(particles[...,0])
        new_p.append(particles[...,1])
        new_p.append(particles[...,2] / mag)
        new_p.append(particles[...,3] / mag)
        new_p = torch.stack(new_p, dim=-1)

        return new_p
        # return particles


class SyntheticDiskTrackingTransformer(IdentityTransformer):
    def __init__(self):
        pass

class LasotTransformer(IdentityTransformer):
    def __init__(self):
        pass

class UAV123Transformer(IdentityTransformer):
    def __init__(self):
        pass





class House3DTransformer(IdentityTransformer):
    def __init__(self):
        pass

    def forward_tranform(self, particles):

        assert(particles.shape[-1] == 4)

        new_p = []
        new_p.append(particles[...,0])
        new_p.append(particles[...,1])
        new_p.append(torch.atan2(particles[...,2], particles[...,3]))
        new_p = torch.stack(new_p, dim=-1)

        return new_p

    def backward_tranform(self, particles):
        
        assert(particles.shape[-1] == 3)

        new_p = []
        new_p.append(particles[...,0])
        new_p.append(particles[...,1])
        new_p.append(torch.sin(particles[...,2]))
        new_p.append(torch.cos(particles[...,2]))
        new_p = torch.stack(new_p, dim=-1)

        return new_p


    def apply_norm(self, particles):

        if(particles.shape[-1] == 4):

            # Norm the last 2 dims to be 1
            mag = torch.sqrt((particles[...,2] ** 2) + (particles[...,3] ** 2))

            # Add a small eps to prevent div by 0
            mag = mag + 1e-8

            new_p = []
            new_p.append(particles[...,0])
            new_p.append(particles[...,1])
            new_p.append(particles[...,2] / mag)
            new_p.append(particles[...,3] / mag)
            new_p = torch.stack(new_p, dim=-1)
            return new_p

        # elif(particles.shape[-1] == 3):
            

        else:
            # Should never get here?
            # assert(False)
            return particles


    # def downscale(self, p):
    #     assert(p.shape[-1] == 3)

    #     new_p = []
    #     new_p.append(p[...,0] / 100.0)
    #     new_p.append(p[...,1] / 100.0)
    #     new_p.append(p[...,2])
    #     new_p = torch.stack(new_p, dim=-1)

    #     return new_p

    # def upscale(self, p):
    #     assert(p.shape[-1] == 3)

    #     new_p = []
    #     new_p.append(p[...,0] * 100.0)
    #     new_p.append(p[...,1] * 100.0)
    #     new_p.append(p[...,2])
    #     new_p = torch.stack(new_p, dim=-1)

    #     return new_p


    # def downscale_map(self, x, y):
    #     return x/100.0, y/100.0

    # def upscale_map(self, x, y):
    #     return x*100.0, y*100.0
        


def create_particle_transformer(particle_transformer_params):

    assert("transformer_type" in particle_transformer_params.keys())
    transformer_type = particle_transformer_params["transformer_type"]

    if(transformer_type == "Identity"):
        return IdentityTransformer()
    elif(transformer_type == "DeepMindTransformer"):
        return DeepMindTransformer()
    elif(transformer_type == "BearingsOnlyTransformer"):
        return BearingsOnlyTransformer()
    elif(transformer_type == "BearingsOnlyVelocityTransformer"):
        return BearingsOnlyVelocityTransformer()
    elif(transformer_type == "BearingsOnlyVectorAngleTransformer"):
        return BearingsOnlyVectorAngleTransformer()
    elif(transformer_type == "SyntheticDiskTrackingTransformer"):
        return SyntheticDiskTrackingTransformer()
    elif(transformer_type == "LasotTransformer"):
        return LasotTransformer()
    elif(transformer_type == "UAV123Transformer"):
        return UAV123Transformer()
    elif(transformer_type == "House3DTransformer"):
        return House3DTransformer()
    else:
        assert(False)

