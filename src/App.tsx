import { useState } from "react";

import { pyInvoke } from "tauri-plugin-pytauri-api";
import TrainingConfigForm from "./components/TrainingConfigForm";
import "./App.css";

function App() {
    const [greetMsg, setGreetMsg] = useState("");
    const [name, setName] = useState("");
    const [checkbox, setCheckbox] = useState(false);

    async function greet() {
        // setGreetMsg(await pyInvoke("greet", { name }));
        console.log('start');
        try {
            const res = await pyInvoke("process_pdf", { pdf_path: name, use_vlm: checkbox });
            console.log(res);
        } catch (error) {
            console.error("Error processing PDF:", error);
        }
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
                    placeholder="Enter a path to PDF..."
                />
                <input
                    type="checkbox"
                    checked={checkbox}
                    onChange={(e) => setCheckbox(e.target.checked)}
                />
                <button type="submit">转换 PDF</button>
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
