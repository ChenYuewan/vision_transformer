import jax
import jax.numpy as jnp
import flax
from flax.training import checkpoints
from flax.training import common_utils
from flax.training import train_state
import optax
import tensorflow_datasets as tfds
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from vit_jax import models, checkpoint, input_pipeline, utils, train
from vit_jax.configs import common as common_config, models as models_config


# ======================================================
# 1. Load dataset
# ======================================================
dataset = 'cifar10'
batch_size = 512 #一次训练用 512 张图片做一次梯度更新
config = common_config.with_dataset(common_config.get_config(), dataset)
config.batch = batch_size
config.pp.crop = 224

ds_train = input_pipeline.get_data_from_tfds(config=config, mode='train')
ds_test = input_pipeline.get_data_from_tfds(config=config, mode='test')
num_classes = input_pipeline.get_dataset_info(dataset, 'train')['num_classes']

# ======================================================
# 2. Load model
# ======================================================
model_name = 'ViT-B_32'
model_config = models_config.MODEL_CONFIGS[model_name]
print("Model Config:", model_config)

model = models.VisionTransformer(num_classes=num_classes, **model_config)
variables = jax.jit(lambda: model.init(
    jax.random.PRNGKey(0),
    jnp.ones((1, 224, 224, 3)),
    train=False,
))()

# ======================================================
# 3. Load pretrained checkpoint
# ======================================================
params = checkpoint.load_pretrained(
    pretrained_path=f'{model_name}.npz',
    init_params=variables['params'],
    model_config=model_config,
)
print("Loaded pretrained params.")

params_repl = flax.jax_utils.replicate(params)

# ======================================================
# 4. Evaluate (random / pretrained)
# ======================================================
vit_apply_repl = jax.pmap(lambda params, inputs: model.apply(
    dict(params=params), inputs, train=False))

def get_accuracy(params_repl):
    good, total = 0, 0
    steps = input_pipeline.get_dataset_info(dataset, 'test')['num_examples'] // batch_size
    for _, batch in zip(tqdm(range(steps)), ds_test.as_numpy_iterator()):
        predicted = vit_apply_repl(params_repl, batch['image'])
        is_same = predicted.argmax(axis=-1) == batch['label'].argmax(axis=-1)
        good += is_same.sum()
        total += len(is_same.flatten())
    return good / total

acc = get_accuracy(params_repl)
print("Initial (pretrained) accuracy:", acc)


# ======================================================
# 5. Fine-tune
# ======================================================
total_steps = 1000
warmup_steps = 5
decay_type = 'cosine'
grad_norm_clip = 1
accum_steps = 8
base_lr = 0.03

lr_fn = utils.create_learning_rate_schedule(total_steps, base_lr, decay_type, warmup_steps)
tx = optax.chain(
    optax.clip_by_global_norm(grad_norm_clip),
    optax.sgd(learning_rate=lr_fn, momentum=0.9, accumulator_dtype='bfloat16')
)
update_fn = train.make_update_fn(apply_fn=model.apply, accum_steps=accum_steps, tx=tx)
update_fn_repl = flax.jax_utils.replicate(update_fn)
opt_state = tx.init(params)
opt_state_repl = flax.jax_utils.replicate(opt_state)
update_rng_repl = flax.jax_utils.replicate(jax.random.PRNGKey(0))

losses = []
for step, batch in zip(tqdm(range(1, total_steps + 1)), ds_train.as_numpy_iterator()):
    params_repl, opt_state_repl, loss_repl, update_rng_repl = update_fn_repl(
        params_repl, opt_state_repl, batch, update_rng_repl)
    losses.append(loss_repl[0])

plt.plot(losses)
plt.title("Training Loss Curve")
plt.show()

# ======================================================
# 6. Final evaluation
# ======================================================
acc = get_accuracy(params_repl)
print("Final fine-tuned accuracy:", acc)
