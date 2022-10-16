# Unstable Book

## An (unsuccessful) attempt to setup the [Stable Diffusion](https://stability.ai/blog/stable-diffusion-public-release) inference pipeline in Elixir

There are now numerous python implementations of Stable Diffusion inference ([original CompVis release](https://github.com/CompVis/stable-diffusion), [Huggingface Diffusers release](https://huggingface.co/blog/stable_diffusion), [high performance KerasCV version](https://keras.io/guides/keras_cv/generate_images_with_stable_diffusion/)), so I decided to participate in Spawnfest 2022 and see how hard it would be to implement a copy of this in Axon, using livebook as an interactive learning environment.

What followed was a 48 hour experiement on the [Dunning–Kruger effect](https://en.wikipedia.org/wiki/Dunning%E2%80%93Kruger_effect) wherein I came to realise how little I know about machine learning, but struggled through nevertheless 😂.  Read on for the post mortem.

The architecture of the model pipeline is as follows:

![Stable diffusion pipeline architecture](https://i.imgur.com/2uC8rYJ.png)
(image taken from https://keras.io/examples/generative/random_walks_with_stable_diffusion/)

### Tokenizer

The one part of the pipeline that has been implemented isn't shown in the above diagram (🤦‍♂️), but it is the tokenizer that takes the prompt text you have entered and uses [Byte-Pair Encoding](https://towardsdatascience.com/byte-pair-encoding-subword-based-tokenization-algorithm-77828a70bee0) to convert it into a sequence of numerical vectors, to be passed on as the input to the Text Encoder.

The [livebook](tokenizer.livemd ) has the elixir implementation of the Tokenizer.

### ONNX Export

Hugging Face have released a script with their diffusers library to [output ONNX exports of the pipeline parts](https://huggingface.co/blog/diffusers-2nd-month#experimental-onnx-exporter-and-pipeline). 

Running that script outputs these ONNX models:

```
/exported_models/stable_diffusion_onnx
├── text_encoder
│   └── model.onnx
├── unet
│   ├── model.onnx
│   └── weights.pb
├── vae_decoder
└── vae_encoder
    └── model.onnx
```

I wasn't able to get Axon to successfully import one of these models

```elixir
{model, params} =
  AxonOnnx.import(
    "/exported_models/stable_diffusion_onnx/text_encoder/model.onnx"
  )
```

```
** (CaseClauseError) no case clause matching: {:env, [#Function<83.44201105/1 in EXLA.Backend.reshape/2>, EXLA]}
    (exla 0.2.3) lib/exla/defn.ex:303: EXLA.Defn.compile/7
    (exla 0.2.3) lib/exla/defn.ex:224: EXLA.Defn.__jit__/5
    (nx 0.3.0) lib/nx/defn.ex:432: Nx.Defn.do_jit_apply/3
    (axon_onnx 0.2.1) lib/axon_onnx/deserialize.ex:67: anonymous fn/2 in AxonOnnx.Deserialize.get_params/1
    (elixir 1.14.0) lib/enum.ex:2468: Enum."-reduce/3-lists^foldl/2-0-"/3
    (axon_onnx 0.2.1) lib/axon_onnx/deserialize.ex:41: AxonOnnx.Deserialize.graph_to_axon/2
    (axon_onnx 0.2.1) lib/axon_onnx/deserialize.ex:27: AxonOnnx.Deserialize.to_axon/2
```

### Running livebook in Google Collab

In case this EXLA error was caused by running on a M1 mac, I also tried running the livebook on a Google Collab with GPU attached, using [this notebook](livebook_on_colab_with_cloudflare_tunnel.ipynb) to install livebook and then run a Cloudflare Tunnel to enable access. Running from this seperate environment still resulted in the same error. 
