const express = require("express");
const app = express();
const cors = require("cors");
const tf = require("@tensorflow/tfjs-node");

const bodyParser = require("body-parser");
const MongoClient = require("mongodb").MongoClient;

// const model = require("./model");
let model;
const PORT = process.env.PORT || 4000;

app.use(cors());
app.use(express.static("public"));
app.use(bodyParser.json());

const loadModel = async () => {
  // model = await tf.loadGraphModel("file://MultVaeJS(64-32-64)/model.json");
  model = await tf.loadGraphModel(
    "file://VAEModel(JS)(128-64-32-16-8-16-32-64-128)/js/model.json"
  );
  // model = await tf.loadGraphModel("file://MultVaeJS/model.json");
  // model = await tf.loadGraphModel("file://HVAE-EMBEDDING-KDD-JS/js/model.json");
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

const findIndexFromTitle = async (title) => {
  let allValues;

  const titleRegex = new RegExp(title, "i");

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
      { title: titleRegex },
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

app.post("/findIndex", async (req, res) => {
  const title = req.body.title;

  const index = await findIndexFromTitle(title);

  console.log(index);
  res.send({ index: index });
});

const findGenres = async (list) => {
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
        projection: { _id: 0, genres: 1 },
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

    let cursor = collection
      .find({ index: { $in: list } })
      .sort({ popularity: -1 });

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

const findMovieWithGenre = async (list, genre) => {
  const genreRegex = new RegExp(genre, "i");
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

    let cursor = collection
      .find({ index: { $in: list }, genres: genreRegex })
      .sort({ popularity: -1 });

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
    if (
      typeof preferred_movies === "undefined" ||
      preferred_movies.length === 0
    ) {
      res.send(
        "NO PREFERENCE SENT.SEND POST BODY IN FORMAT {'preferred_movies':[0,1,2...]}"
      );
    } else {
      try {
        preferred_movies = preferred_movies.filter((index) => index < 62000);

        const indices_array = tf.tensor1d(preferred_movies, (dtype = "int32"));
        const value = tf.tensor1d(
          preferred_movies.map(() => 1),
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

        let count = 0;
        let movieList = [];
        while (movieList.length <= req.params.k) {
          if (!preferred_movies.includes(movies[count]))
            movieList.push(movies[count]);

          count++;
        }
        list = await findMovie(movieList);

        res.send({ movie: list });
      } catch (error) {
        res.send({ error });
      }
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
    // indices = indices.map(String);
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

    // movies = movies.map(String);
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

const defaultPredictGenre = async (req, res) => {
  const k = Math.min(req.params.k, 50);
  console.log(req.params.genre);
  if (typeof model !== "undefined") {
    let indices = [3147, 1721, 1, 2028, 50, 527, 608];
    indices.sort((a, b) => a - b);
    // indices = indices.map(String);
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

    input.dispose();
    result.dispose();

    let list = [];
    let mainLoopCounter = 0;
    const increment = 100;
    while (mainLoopCounter < 100) {
      const moviesSlice = movies.slice(
        mainLoopCounter,
        mainLoopCounter + increment
      );
      mainLoopCounter += increment;
      console.log("Main LOOP COUNTER: " + mainLoopCounter);
      const movieDetails = await findMovieWithGenre(
        moviesSlice,
        req.params.genre
      );
      // console.log(movieDetails);
      // const movieDetails1 = await findMovie(moviesSlice);
      // console.log(movieDetails1);

      if (
        !movieDetails ||
        movieDetails?.length === 0 ||
        typeof movieDetails == "undefined"
      ) {
        continue;
      }

      // console.log(movieGenres);
      // console.log(movieGenres[0]?.genres.toLowerCase().split("|"));
      // console.log(movieGenres[0]?.genres.split("|")?.includes(req.params.genre));
      // movies = movies.map(String);
      let count = 0;
      while (list.length < req.params.k && count < moviesSlice.length) {
        if (!indices.includes(movieDetails[count]?.index)) {
          if (
            movieDetails[count]?.genres
              ?.toLowerCase()
              .split("|")
              .includes(req.params.genre)
          ) {
            list.push(movieDetails[count]);
          }
        }
        count++;
      }
    }
    // console.log(movies);
    return list;
  } else {
    res.send("MODEL NOT LOADED");
  }

  res.end();
};
app.get("/predict/:genre/:k", async (req, res) => {
  const list = await defaultPredictGenre(req, res);

  res.send({
    movies: list,
    msg:
      "NO PREFERENCE SENT.SEND POST REQUEST WITH BODY IN FORMAT {'preferred_movies':[0,1,2,...<Preffered Movie list>]}",
  });
});

// app.post("/predict/genre/:k", async (req, res) => {
//   let genres = req.body.genres;

//   let outputList = {};
//   Promise.all(
//     genres.map(async (genre) => {
//       req.params.genre = genre;

//       const list = await defaultPredictGenre(req, res);
//       outputList[genre] = list;
//     })
//   ).then(() => {
//     res.send(outputList);
//   });
// });

app.post("/predict/genre/:k", async (req, res) => {
  if (typeof model !== "undefined") {
    let genres = req.body.genres;
    console.log(genres);
    let preferred_movies = req.body.preferred_movies;
    preferred_movies.sort((a, b) => a - b);
    if (
      typeof preferred_movies === "undefined" ||
      preferred_movies.length === 0
    ) {
      res.send(
        "NO PREFERENCE SENT.SEND POST BODY IN FORMAT {'preferred_movies':[0,1,2...]}"
      );
    } else {
      try {
        preferred_movies = preferred_movies.filter((index) => index < 62000);

        const indices_array = tf.tensor1d(preferred_movies, (dtype = "int32"));
        const value = tf.tensor1d(
          preferred_movies.map(() => 1),
          (dtype = "float32")
        );
        const shape = [62000];
        const input = tf.expandDims(
          tf.sparseToDense(indices_array, value, shape),
          0
        );

        const result = await model.predict(input);

        const data = await result.data();

        // Sort Descending
        let movies = [...Array(62000).keys()].sort((a, b) => {
          return data[b] - data[a];
        });

        result.dispose();
        input.dispose();

        let list = {};
        let outputList = {};
        genres.map((genre) => {
          list[genre] = [];
          outputList[genre] = [];
        });

        let mainLoopCounter = 0;
        const increment = 100;
        while (mainLoopCounter <= 100) {
          const moviesSlice = movies.slice(
            mainLoopCounter,
            mainLoopCounter + increment
          );
          mainLoopCounter += increment;
          console.log("Main LOOP COUNTER: " + mainLoopCounter);
          const movieDetails = await findMovie(moviesSlice);

          // console.log(movieGenres);
          // console.log(movieGenres[0]?.genres.toLowerCase().split("|"));
          // console.log(movieGenres[0]?.genres.split("|")?.includes(req.params.genre));
          // movies = movies.map(String);

          if (
            !movieDetails ||
            movieDetails?.length === 0 ||
            typeof movieDetails == "undefined"
          ) {
            continue;
          }
          // console.log(movieDetails);

          let count = 0;
          while (count < moviesSlice.length) {
            // console.log("here");
            if (!preferred_movies.includes(movieDetails[count]?.index)) {
              const movieGenres = movieDetails[count]?.genres
                ?.toLowerCase()
                .split("|");
              // console.log("MOVIE GENRES");
              // console.log(movieGenres);
              Object.keys(list).map((genre) => {
                // console.log(genre);
                if (movieGenres.includes(genre)) {
                  list[genre].push(movieDetails[count]);
                }
              });
            }
            count++;
          }

          Object.keys(list).map((genre) => {
            if (list[genre].length >= req.params.k) {
              outputList[genre] = list[genre];
              delete list[genre];
            }
          });
          if (Object.keys(list).length <= 0) {
            break;
          }
        }

        res.send(outputList);
      } catch (error) {
        let genres = req.body.genres;

        let outputList = {};
        Promise.all(
          genres.map(async (genre) => {
            req.params.genre = genre;

            const list = await defaultPredictGenre(req, res);
            outputList[genre] = list;
          })
        ).then(() => {
          res.send(outputList);
        });
      }
    }
  } else {
    res.send("MODEL NOT LOADED");
  }

  res.end();
});

app.post("/predict/:genre/:k", async (req, res) => {
  if (typeof model !== "undefined") {
    let preferred_movies = req.body.preferred_movies;
    preferred_movies.sort((a, b) => a - b);
    if (
      typeof preferred_movies === "undefined" ||
      preferred_movies.length === 0
    ) {
      res.send(
        "NO PREFERENCE SENT.SEND POST BODY IN FORMAT {'preferred_movies':[0,1,2...]}"
      );
    } else {
      try {
        preferred_movies = preferred_movies.filter((index) => index < 62000);
        // console.log(preferred_movies);
        const indices_array = tf.tensor1d(preferred_movies, (dtype = "int32"));
        const value = tf.tensor1d(
          preferred_movies.map(() => 1),
          (dtype = "float32")
        );
        const shape = [62000];
        const input = tf.expandDims(
          tf.sparseToDense(indices_array, value, shape),
          0
        );

        const result = await model.predict(input);

        const data = await result.data();

        // Sort Descending
        let movies = [...Array(62000).keys()].sort((a, b) => {
          return data[b] - data[a];
        });

        result.dispose();
        input.dispose();

        let list = [];
        let mainLoopCounter = 0;
        const increment = 50;
        while (list.length < req.params.k && mainLoopCounter < 1000) {
          const moviesSlice = movies.slice(
            mainLoopCounter,
            mainLoopCounter + increment
          );
          mainLoopCounter += increment;
          console.log("Main LOOP COUNTER: " + mainLoopCounter);
          const movieDetails = await findMovieWithGenre(
            moviesSlice,
            req.params.genre
          );
          // console.log(movieDetails);
          // console.log(movieGenres);
          // console.log(movieGenres[0]?.genres.toLowerCase().split("|"));
          // console.log(movieGenres[0]?.genres.split("|")?.includes(req.params.genre));
          // movies = movies.map(String);

          if (
            !movieDetails ||
            movieDetails?.length === 0 ||
            typeof movieDetails == "undefined"
          ) {
            continue;
          }

          let count = 0;
          while (list.length < req.params.k && count < moviesSlice.length) {
            if (!preferred_movies.includes(movieDetails[count]?.index)) {
              if (
                movieDetails[count]?.genres
                  ?.toLowerCase()
                  .split("|")
                  .includes(req.params.genre)
              ) {
                list.push(movieDetails[count]);
              }
            }
            count++;
          }
        }

        // console.log(list);

        list = list.sort((a, b) => {
          return b.popularity - a.popularity;
        });
        // console.log(list);

        res.send({ movie: list });
      } catch (error) {
        const list = await defaultPredictGenre(req, res);

        res.send({
          movies: list,
        });
      }
    }
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
app.get("/", (req, res) => {
  res.send({
    msg:
      "This api is open at /predict/k where k is number of recommendation required like /predict/5 ",
  });
});
404;
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
