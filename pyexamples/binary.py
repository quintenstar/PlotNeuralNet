#%%
import sys

sys.path.append("../")
from pycore.tikzeng import *
from pycore.blocks import *

fname = "binary.tex"
inputs = [
    "../inputs/sdf_to_binary.png",
    # "../inputs/sdf.png",
    # "../inputs/sdf_to_dist_x.png",
    # "../inputs/sdf_to_dist_y.png",
    # "../inputs/pressure.png",
    # "../inputs/velocity.png",
    # "../inputs/velocity.png",
]
# print(len(inputs))

N_channels = len(inputs)
Nx = 256
Ny = 128
N_conv = 6
filter = 8

N_linear = 2
linear_size = 512
linear_factor = 4

batch_norm = True


# visual
filter_max = 64
scaling = 4
lin_size = 256 // scaling

conv_layers = []
for i, input in enumerate(inputs):
    # print(i)
    conv_layers.append(
        to_input(
            input,
            to=f"({-len(inputs)+i},0,0)",
            width=Nx // scaling // 4,
            height=Ny // scaling // 4,
        )
    )


to = "(0,0,0)"
for i in range(N_conv):

    conv = f"conv{i+1}"
    pool = f"pool{i+1}"
    batch = f"batch{i+1}"

    conv_layers.append(
        to_ConvRelu(
            conv,
            x=Nx,
            y=Ny,
            n_filer=filter,
            offset=f"(2,0,0)",
            to=to,
            width=max(filter, filter_max) // scaling // (i + 1),
            height=Ny // scaling,
            depth=Nx // scaling,
            caption=conv,  # f"{conv} + ReLU {'+ Batchnorm' if batch_norm else ''}+ Pool",
        )
    )

    to = f"({conv}-east)"

    if batch_norm is True:
        conv_layers.append(
            to_BN(
                batch,
                offset="(0,0,0)",
                to=to,
                width=1,
                height=Ny // scaling,
                depth=Nx // scaling,
            )
        )

        to = f"({batch}-east)"

    if i != 0:
        conv_layers.append(to_connection(f"pool{i}", f"conv{i+1}"))

    Nx = Nx // 2
    Ny = Ny // 2

    conv_layers.append(
        to_Pool(
            pool,
            x="",
            y="",
            n_filer="",
            offset="(0,0,0)",
            to=to,
            width=1,  # max(filter, filter_max) // scaling,
            height=Ny // scaling,
            depth=Nx // scaling,
            opacity=0.5,
            caption="",  # pool,
        ),
    )

    to = f"(pool{i+1}-east)"
    filter = 2 * filter

print(Nx)
print(Ny)
print(filter)

flatten_size = Nx * Ny * filter // 2
conv_layers.append(
    to_Flatten(
        "flatten_1",
        "z",
        flatten_size,
        offset="(2.5,0,0)",
        to=to,
        width=1,
        height=1,
        depth=lin_size,
        caption="flatten",
    )
)

conv_layers.append(to_connection(f"pool{i+1}", "flatten_1"))

to = "(flatten_1-east)"


linear_layers = []

for i in range(N_linear):
    hidden = f"hidden{i+1}"
    linear_layers.append(
        to_Flatten(
            hidden,
            "z",
            linear_size,
            offset="(1.5,0,0)",
            to=to,
            width=1,
            height=1,
            depth=int(lin_size * (linear_size / flatten_size)),
            caption=hidden,
        )
    )

    if i == 0:
        linear_layers.append(to_connection(f"flatten_1", hidden))
    else:
        linear_layers.append(to_connection(f"hidden{i}", hidden))

    to = f"(hidden{i+1}-east)"

    linear_size = linear_size // linear_factor


linear_layers.append(
    to_Flatten(
        "output",
        "z",
        1,
        offset="(1.5,0,0)",
        to=to,
        width=1,
        height=1,
        depth=1,
        caption="Output",
    )
)

if N_linear == 0:
    linear_layers.append(to_connection(f"flatten_1", "output"))
else:
    linear_layers.append(to_connection(f"hidden{i+1}", "output"))


arch = [
    to_head(".."),
    to_cor(),
    to_begin(),
    # input
    *conv_layers,
    *linear_layers,
    # to_Flatten("test"),
    to_end(),
]


# print(arch)


def main():
    # namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, fname)


if __name__ == "__main__":
    main()
