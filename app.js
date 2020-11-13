const express = require("express");
const app = express();
const cors = require("cors");
const tf = require("@tensorflow/tfjs-node");

const PORT = process.env.PORT || 5000;

app.use(cors());
app.use(express.static("public"));

app.get("/hello", async (req, res) => {
  const model = await tf.loadGraphModel("file://multvae/model.json");

  let zeros = (w, h, v = 0) =>
    Array.from(new Array(h), (_) => Array(w).fill(v));

  const arr = zeros(62000, 1, 0);
  console.log(arr);
  const tensor = tf.tensor(arr);
  const result = await model.predict(tensor);
  console.log(result);
  console.log("hello");
  res.send("hello world!");
});

app.listen(PORT, () => {
  console.log(`Example app listening at http://localhost:${PORT}`);
});
