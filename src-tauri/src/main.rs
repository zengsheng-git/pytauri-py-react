// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
  // Get the path to the Resources directory
  let exe_path = std::env::current_exe()
    .expect("Failed to get current executable path");
  
  let resources_dir = exe_path
    .parent()
    .expect("Failed to get parent directory")
    .parent()
    .expect("Failed to get grandparent directory")
    .join("Resources")
    .join("_up_");
  
  // Execute Python main.py with the correct path
  let status = std::process::Command::new("python3")
    .arg(resources_dir.join("main.py"))
    .current_dir(&resources_dir)
    .status();
  
  match status {
    Ok(exit_status) => {
      if !exit_status.success() {
        eprintln!("Python process exited with status: {}", exit_status);
      }
    },
    Err(e) => {
      eprintln!("Failed to execute main.py: {}", e);
      eprintln!("Resources directory: {}", resources_dir.display());
      eprintln!("Current directory: {}", std::env::current_dir().unwrap_or_default().display());
    }
  }
}
