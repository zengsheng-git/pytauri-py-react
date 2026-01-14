import { useState } from "react";
import { Input, Button, Switch } from "@pikoloo/darwin-ui";
import { pyInvoke } from "tauri-plugin-pytauri-api";

export default () => {
  const [greetMsg, setGreetMsg] = useState("");
  const [name, setName] = useState("");
  const [checkbox, setCheckbox] = useState(false);

  async function greet() {
    // setGreetMsg(await pyInvoke("greet", { name }));
    console.log("start");
    try {
      const res = await pyInvoke("process_pdf", {
        pdf_path: name,
        use_vlm: checkbox,
      });
      console.log(res);
    } catch (error) {
      console.error("Error processing PDF:", error);
    }
  }

  return (
    <>
      <form
        className="flex gap-5 w-[400px]"
        onSubmit={(e) => {
          e.preventDefault();
          greet();
        }}
      >
        <Input
          id="greet-input"
          onChange={(e) => setName(e.currentTarget.value)}
          placeholder="Enter a path to PDF..."
        />
        <Switch checked={checkbox} onChange={(e) => setCheckbox(e)} />
        <Button variant="primary" type="submit">
          转换 PDF
        </Button>
      </form>
      <p>{greetMsg}</p>
    </>
  );
};
