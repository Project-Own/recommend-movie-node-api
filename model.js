const tf = require("@tensorflow/tfjs-node");

const model_encoder = (input_dim, latent_dim, dims, batch_size = 1) => {
  encoder_inputs = tf.input({ shape: [input_dim], name: "input_encoder" });
  x = tf.layers
    .dense({
      units: dims[0],
      activation: "tanh",
      kernelInitializer: tf.initializers.glorotUniform(),
      biasInitializer: tf.initializers.truncatedNormal({ stddev: 0.001 }),
      kernelRegularizer: tf.regularizers.l2(),
      name: "Dense_1",
    })
    .apply(encoder_inputs);
  z_mean = tf.layers.dense({ units: latent_dim, name: "z_mean" }).apply(x);
  z_log_var = tf.layers
    .dense({ units: latent_dim, name: "z_log_var" })
    .apply(x);
  encoder = tf.model({
    inputs: encoder_inputs,
    outputs: [z_log_var, z_mean],
  });
  return encoder;
};

const model_decoder = (input_dim, latent_dim, dims) => {
  latent_inputs = tf.input({ shape: [latent_dim], name: "LATENT_DECODER" });
  x = tf.layers
    .dense({
      units: dims[0],
      activation: "tanh",
      kernelInitializer: tf.initializers.glorotUniform(),
      biasInitializer: tf.initializers.truncatedNormal({ stddev: 0.001 }),
    })
    .apply(latent_inputs);
  decoder_outputs = tf.layers
    .dense({
      units: input_dim,
      // activation: "tanh",
      kernelInitializer: tf.initializers.glorotUniform(),
      biasInitializer: tf.initializers.truncatedNormal({ stddev: 0.001 }),
    })
    .apply(x);
  decoder = tf.model({
    inputs: latent_inputs,
    outputs: decoder_outputs,
    name: "DECODER",
  });
  return decoder;
};

class MULTVAE extends tf.LayersModel {
  constructor(
    encoder,
    decoder,
    lam = 0.03,
    total_anneal_steps = 200000,
    anneal_cap = 0.2,
    config = {}
  ) {
    super({});
    this.encoder = encoder;
    this.decoder = decoder;
    this.lam = lam;
    this.total_anneal_steps = total_anneal_steps;
    this.anneal_cap = anneal_cap;
    this.update_count = 0;
  }
  call(inputs) {
    output = this.encoder(inputs);
    z_mean = output[0];
    z_log_var = output[1];
    z =
      z_mean +
      tf.exp(0.5 * z_log_var) * tf.randomNormal([batch_size, latent_dim]);

    logits = this.decoder(z);

    kl_loss = tf.mean(
      -0.5 *
        tf.sum(
          z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1,
          (axis = 1)
        )
    );

    if (this.total_anneal_steps > 0) {
      anneal = min(
        this.anneal_cap,
        1 * (this.update_count / this.total_anneal_steps)
      );
    } else {
      anneal = this.anneal_cap;
    }

    loss = tf.mean(
      tf.sum(tf.losses.sigmoidCrossEntropy(inputs, logits), (axis = 1))
    );

    reg_loss = 2 * tf.sum(this.losses);

    loss = loss + this.lam * reg_loss + anneal * kl_loss;
    this.addLoss(loss);
    return logits;
  }
}
module.exports = {
  encoder: model_encoder,
  decoder: model_decoder,
  vae: MULTVAE,
};
