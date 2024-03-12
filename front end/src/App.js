import React from "react";
// import {useState, useEffect} from "react";
// import { BrowserRouter as Router, Route, Link, NavLink, Switch, Routes } from 'react-router-dom';
import "./App.css";
import ImageUpload from './components/ImageUpload';

function App() 
{

// const [data, setData] = useState([{}])

// useEffect( () => {
//   fetch("/").then (
//     res => res.json()
//   ).then(
//     data => {
//       setData(data)
//       console.log(data)
//     }
//   )
// }, [])

// const [predictedResult, setPredictedResult] = useState(null);
// const handlePrediction = (prediction) => {
//   setPredictedResult(prediction);
// }

  return (
    <div className="container">
    <section className="background"> </section>
      <div>
        <h1 className="Title"> Rice Plant Disease Detection System</h1>
      </div> 
      

      <div className="overlay">
      <ImageUpload />
      </div>
      </div>
  );
};

export default App;
