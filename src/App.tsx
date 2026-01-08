import { useState } from "react";
import reactLogo from "./assets/react.svg";
import { pyInvoke } from "tauri-plugin-pytauri-api";
import TrainingConfigForm from "./components/TrainingConfigForm";
import "./App.css";

function App() {
    const [greetMsg, setGreetMsg] = useState("");
    const [name, setName] = useState("");

    async function greet() {
        setGreetMsg(await pyInvoke("greet", { name }));
    }

    return (
        <main className="container">
            <form
                className="row"
                onSubmit={(e) => {
                    e.preventDefault();
                    greet();
                }}
            >
                <input
                    id="greet-input"
                    onChange={(e) => setName(e.currentTarget.value)}
                    placeholder="Enter a name..."
                />
                <button type="submit">测试调用 Python</button>
            </form>
            <p>{greetMsg}</p>

            <div className="training-section">
                <h2>MLX-LM-LoRA Training Configuration</h2>
                <TrainingConfigForm />
            </div>
        </main>
    );
}

export default App;
