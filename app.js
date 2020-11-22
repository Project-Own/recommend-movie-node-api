const express = require("express");
const app = express();
const cors = require("cors");
const tf = require("@tensorflow/tfjs-node");

const bodyParser = require("body-parser");
const MongoClient = require("mongodb").MongoClient;

// const model = require("./model");
let model;
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.static("public"));
app.use(bodyParser.json());

const loadModel = async () => {
  model = await tf.loadGraphModel("file://MultVaeJS/model.json");
  console.log("MODEL LOADED");
};

loadModel();

const uri =
  "mongodb+srv://nirjal123:nirjal123@cluster0.6jptv.mongodb.net/Movie?retryWrites=true&w=majority";
const client = new MongoClient(uri, {
  useNewUrlParser: true,
  useUnifiedTopology: true,
});

const findIndex = async (list) => {
  let allValues;
  const client = await MongoClient.connect(uri, {
    useNewUrlParser: true,
    useUnifiedTopology: true,
  }).catch((err) => console.log(err));

  if (!client) {
    return;
  }

  try {
    const db = client.db("Movie");
    const collection = db.collection("Movie");

    let cursor = collection.find(
      { movieId: { $in: list } },
      {
        projection: { _id: 0, index: 1 },
      }
    );

    // for await (const doc of cursor) {
    //   console.log(doc);
    // }
    allValues = await cursor.toArray();

    // console.log(allValues);
  } catch (err) {
    console.log(err);
  } finally {
    client.close();
  }
  return allValues;
};

const findMovie = async (list) => {
  let allValues;
  const client = await MongoClient.connect(uri, {
    useNewUrlParser: true,
    useUnifiedTopology: true,
  }).catch((err) => console.log(err));

  if (!client) {
    return;
  }

  try {
    const db = client.db("Movie");
    const collection = db.collection("Movie");

    let cursor = collection.find(
      { index: { $in: list } },
      {
        projection: {
          _id: 0,
          genres: 1,
          title: 1,
          posterPath: 1,
          index: 1,
          movieId: 1,
        },
      }
    );

    // for await (const doc of cursor) {
    //   console.log(doc);
    // }
    allValues = await cursor.toArray();

    // console.log(allValues);
  } catch (err) {
    console.log(err);
  } finally {
    client.close();
  }
  return allValues;
};

// getData(["0"])
app.post("/predict/:k", async (req, res) => {
  if (typeof model !== "undefined") {
    preferred_movies = req.body.preferred_movies;
    preferred_movies.sort((a, b) => a - b);
    preferred_movies = preferred_movies.map(String);
    if (
      typeof preferred_movies === "undefined" ||
      preferred_movies.length === 0
    ) {
      res.send(
        "NO PREFERENCE SENT.SEND POST BODY IN FORMAT {'preferred_movies':[0,1,2...]}"
      );
    } else {
      let indices = await findIndex(preferred_movies);

      indices = indices.map((value) => value.index);
      const indices_array = tf.tensor1d(
        indices.map((index) => index),
        (dtype = "int32")
      );
      const value = tf.tensor1d(
        indices.map(() => 1),
        (dtype = "float32")
      );
      const shape = [62000];
      const input = tf.expandDims(
        tf.sparseToDense(indices_array, value, shape),
        0
      );

      const result = await model.predict(input);

      const data = await result.data();

      result.dispose();
      input.dispose();

      // Sort Descending
      let movies = [...Array(62000).keys()].sort((a, b) => {
        return data[b] - data[a];
      });

      movies = movies.map(String);
      let count = 0;
      let movieList = [];
      while (movieList.length <= req.params.k) {
        if (!indices.includes(movies[count])) movieList.push(movies[count]);

        count++;
      }
      list = await findMovie(movieList);

      res.send({ movie: list });
    }
  } else {
    res.send("MODEL NOT LOADED");
  }

  res.end();
});

app.get("/predict/:k", async (req, res) => {
  if (typeof model !== "undefined") {
    let indices = [3147, 1721, 1, 2028, 50, 527, 608];
    indices.sort((a, b) => a - b);
    indices = indices.map(String);
    // console.log(indices);
    indices = await findIndex(indices);
    // console.log(indices);
    indices = indices.map((value) => value.index);
    // console.log(indices);
    const indices_array = tf.tensor1d(
      indices.map((index) => index),
      (dtype = "int32")
    );
    const value = tf.tensor1d(
      indices.map(() => 1),
      (dtype = "float32")
    );
    const shape = [62000];
    const input = tf.expandDims(
      tf.sparseToDense(indices_array, value, shape),
      0
    );

    const result = await model.predict(input);
    const data = await result.data();

    // descending sort
    // data.sort(function (a, b) {
    //   return b - a;
    // });

    let movies = [...Array(62000).keys()].sort((a, b) => {
      return data[b] - data[a];
    });

    movies = movies.map(String);
    let count = 0;
    let movieList = [];
    while (movieList.length <= req.params.k) {
      if (!indices.includes(movies[count])) {
        movieList.push(movies[count]);
      }
      count++;
    }
    // console.log(movieList);
    list = await findMovie(movieList);
    // console.log(movies);
    // console.log(list);
    input.dispose();
    result.dispose();

    res.send({
      movies: list,
      msg:
        "NO PREFERENCE SENT.SEND POST REQUEST WITH BODY IN FORMAT {'preferred_movies':[0,1,2,...<Preffered Movie list>]}",
    });
  } else {
    res.send("MODEL NOT LOADED");
  }

  res.end();
});

// app.get("/model", (req, res) => {
//   input_dim = 62000;
//   latent_dim = 32;
//   encoder_dims = [128];

//   let indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
//   let indices_array = tf.tensor1d(
//     indices.map((index) => index),
//     (dtype = "int32")
//   );
//   let value = tf.tensor1d(
//     indices.map(() => 1),
//     (dtype = "float32")
//   );

//   let shape = [62000];
//   let dense = tf.sparseToDense(indices_array, value, shape);
//   let denseExpanded = tf.expandDims(dense, 0);

//   encoder = model.encoder(input_dim, latent_dim, encoder_dims);
//   decoder = model.decoder(input_dim, latent_dim, encoder_dims);
//   vae = new model.vae(encoder, decoder);
//   output = encoder.predict(denseExpanded);
//   z_mean = output[0];
//   z_log_var = output[1];
//   z = tf.add(
//     z_mean,
//     tf.mul(tf.exp(tf.mul(0.5, z_log_var)), tf.randomNormal([1, latent_dim]))
//   );

//   decoded_output = decoder.predict(z);
//   console.log(z);
//   console.log(decoded_output);
//   console.log("hello");
//   res.send("Model");
// });

app.get("*", (req, res) => {
  res.send("404");
  res.end();
});
app.post("*", (req, res) => {
  res.send("404");
  res.end();
});
app.listen(PORT, () => {
  console.log(`Example app listening at http://localhost:${PORT}`);
});
