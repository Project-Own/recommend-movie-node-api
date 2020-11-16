const express = require("express");
const app = express();
const cors = require("cors");
const tf = require("@tensorflow/tfjs-node");
const float16 = require("@petamoriken/float16");
const model = require("./model");

const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.static("public"));

app.get("/hello", async (req, res) => {
  const model = await tf.loadGraphModel("file://MultVaeJS/model.json");
  let zeros = (w, h, v = 0) =>
    Array.from(new Array(h), (_) => Array(w).fill(v));
  let indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
  let indices_array = tf.tensor1d(
    indices.map((index) => index),
    (dtype = "int32")
  );
  let value = tf.tensor1d(
    indices.map(() => 1),
    (dtype = "float32")
  );
  let shape = [62000];
  let dense = tf.sparseToDense(indices_array, value, shape);
  let denseExpanded = tf.expandDims(dense, 0);

  console.log(indices_array);
  console.log(value);
  console.log(denseExpanded);
  console.log(model.inputs);
  // console.log(tf.sparseToDen/zse(indices_array, value));
  // const result = await model.executeAsync(tf.zeros([1, 62000]));
  const result = await model.predict(denseExpanded);
  console.log(result);
  console.log("hello");
  res.send("hello world!");
});

app.get("/model", (req, res) => {
  input_dim = 62000;
  latent_dim = 32;
  encoder_dims = [128];

  let indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
  let indices_array = tf.tensor1d(
    indices.map((index) => index),
    (dtype = "int32")
  );
  let value = tf.tensor1d(
    indices.map(() => 1),
    (dtype = "float32")
  );

  let shape = [62000];
  let dense = tf.sparseToDense(indices_array, value, shape);
  let denseExpanded = tf.expandDims(dense, 0);

  encoder = model.encoder(input_dim, latent_dim, encoder_dims);
  decoder = model.decoder(input_dim, latent_dim, encoder_dims);
  vae = new model.vae(encoder, decoder);
  output = encoder.predict(denseExpanded);
  z_mean = output[0];
  z_log_var = output[1];
  z = tf.add(
    z_mean,
    tf.mul(tf.exp(tf.mul(0.5, z_log_var)), tf.randomNormal([1, latent_dim]))
  );

  decoded_output = decoder.predict(z);
  console.log(z);
  console.log(decoded_output);
  console.log("hello");
  res.send("Model");
});
app.listen(PORT, () => {
  console.log(`Example app listening at http://localhost:${PORT}`);
});
