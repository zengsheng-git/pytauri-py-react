import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
// import '@pikoloo/darwin-ui/styles';
// import '../index.css'
import "../index1.css";
import "./resetUI.scss";

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
