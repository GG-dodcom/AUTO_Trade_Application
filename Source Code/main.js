console.log(`Electron is using Node.js version: ${process.versions.node}`);

console.log("Loading electron...");
const { app, BrowserWindow } = require("electron");

// Enforce single instance
const gotTheLock = app.requestSingleInstanceLock();

if (!gotTheLock) {
	console.log("Another instance is already running. Exiting...");
	app.quit();
	process.exit(0); // Force immediate exit
}

// Load other modules only if lock is acquired
console.log("Loading path...");
const path = require("path");

console.log("Loading dotenv...");
require("dotenv").config({ path: path.join(app.getAppPath(), ".env") }); // Now app is defined

console.log("Loading child_process...");
const { spawn, exec } = require("child_process");

console.log("Loading next...");
const next = require("next");

console.log("Loading http...");
const http = require("http");

console.log("Loading wait-on...");
const waitOn = require("wait-on");
const net = require("net");

// Rest of your existing code
let pythonProcess;
let serverProcess;
let win;

app.on("second-instance", () => {
	console.log(
		"Second instance attempted to start. Focusing existing window..."
	);
	if (win) {
		if (win.isMinimized()) win.restore();
		win.focus();
	}
	console.log("Python process running:", pythonProcess ? "Yes" : "No");
	console.log("WebSocket server running:", serverProcess ? "Yes" : "No");
});

const env = app.isPackaged
	? "production"
	: process.env.NODE_ENV || "development";
const dev = env === "development";
console.log(`Running in ${dev ? "development" : "production"} mode`);

const nextApp = next({
	dev,
	dir: __dirname,
});
const handle = nextApp.getRequestHandler();

// Function to forcefully terminate a process and its children on Windows
const terminateProcess = (process, callback) => {
	if (!process || !process.pid) {
		console.log("No process to terminate or process already terminated.");
		if (callback) callback();
		return;
	}

	console.log(`Terminating process with PID: ${process.pid}`);
	if (process.platform === "win32") {
		// Use taskkill to terminate the process and its child processes
		exec(`taskkill /PID ${process.pid} /T /F`, (err, stdout, stderr) => {
			if (err) {
				console.error(`Error terminating process ${process.pid}:`, err.message);
				console.error("taskkill stderr:", stderr);
			} else {
				console.log(`Process ${process.pid} terminated successfully`);
			}
			process.pid = null; // Clear the PID to indicate the process is terminated
			if (callback) callback();
		});
	} else {
		// On non-Windows, use SIGKILL
		process.kill("SIGKILL");
		process.pid = null;
		if (callback) callback();
	}
};

// Function to wait for a process to fully terminate
const waitForProcessTermination = (process, timeout = 5000) => {
	return new Promise((resolve) => {
		if (!process || !process.pid) {
			resolve();
			return;
		}

		const checkInterval = 500;
		let elapsed = 0;

		const interval = setInterval(() => {
			exec(`tasklist | findstr ${process.pid}`, (err, stdout) => {
				if (err || !stdout) {
					// Process is no longer running
					clearInterval(interval);
					resolve();
				} else {
					elapsed += checkInterval;
					if (elapsed >= timeout) {
						console.warn(
							`Process ${process.pid} did not terminate within ${timeout}ms`
						);
						clearInterval(interval);
						resolve();
					}
				}
			});
		}, checkInterval);
	});
};

const checkPort = (port) => {
	return new Promise((resolve) => {
		const server = net.createServer();
		server.once("error", (err) => {
			if (err.code === "EADDRINUSE") {
				resolve(false);
			} else {
				console.error(`Port check error on ${port}:`, err.message);
				resolve(false);
			}
		});
		server.once("listening", () => {
			server.close();
			resolve(true);
		});
		server.listen(port);
	});
};

async function createWindow() {
	try {
		// Set working directory explicitly
		const appPath = app.isPackaged
			? path.join(process.resourcesPath, "app")
			: __dirname;
		process.chdir(appPath); // Set working directory
		console.log(`Working directory set to: ${process.cwd()}`); // Log the default working directory
		// const appPath = dev ? __dirname : app.getAppPath();

		console.log("Preparing Next.js app...");
		await nextApp.prepare();
		console.log("Next.js app prepared successfully");

		win = new BrowserWindow({
			width: 1060,
			height: 800,
			webPreferences: {
				preload: path.join(__dirname, "preload.js"),
				nodeIntegration: false,
				contextIsolation: true,
			},
		});

		// Determine paths based on packaged or dev environment
		const pythonPath = dev
			? path.join(appPath, "backend/.venv/Scripts/python.exe")
			: path.join(appPath, "backend/python/python.exe"); // Packaged Python
		const backendScript = path.join(appPath, "backend/app.py");
		const serverScript = path.join(appPath, "server.js");

		// Path to Node.js binary
		const nodePath = dev ? "node" : path.join(appPath, "node", "node.exe");

		// Check if port 4000 is free
		const isPortFree = await checkPort(4000);
		if (!isPortFree) {
			console.error(
				"Port 4000 is already in use. Cannot start WebSocket server."
			);
			app.quit();
			return;
		}

		// Start Python backend
		console.log(`Spawning Python backend: ${pythonPath} ${backendScript}`);
		pythonProcess = spawn(pythonPath, [backendScript], {
			stdio: "inherit",
			cwd: appPath, // Changed from path.dirname(pythonPath) to avoid locking the python folder
			shell: false, // Explicitly disable shell
		});

		pythonProcess.on("spawn", () => console.log("Python backend spawned"));
		pythonProcess.on("error", (err) => {
			console.error("Python spawn error:", err.message);
			if (err.code === "ENOENT") {
				console.error(`Python executable not found at: ${pythonPath}`);
			}
		});
		pythonProcess.on("exit", (code) => {
			console.log(`Python exited with code ${code}`);
			if (code !== 0) {
				console.error(`Python backend crashed. Exit code: ${code}`);
			}
			pythonProcess = null;
		});

		// Start WebSocket server with detailed logging
		console.log(`Spawning WebSocket server: ${nodePath} ${serverScript}`);
		serverProcess = spawn(nodePath, [serverScript], {
			stdio: ["ignore", "pipe", "pipe"], // Capture stdout and stderr
			cwd: appPath,
			shell: false, // Explicitly disable shell
		});

		// Log server output
		serverProcess.stdout.on("data", (data) => {
			console.log(`WebSocket server stdout: ${data}`);
		});
		serverProcess.stderr.on("data", (data) => {
			console.error(`WebSocket server stderr: ${data}`);
		});

		serverProcess.on("spawn", () => console.log("WebSocket server spawned"));
		serverProcess.on("error", (err) => {
			console.error("WebSocket server spawn error:", err.message);
			if (err.code === "ENOENT") {
				console.error(`Node executable not found at: ${process.execPath}`);
			}
		});
		serverProcess.on("exit", (code, signal) => {
			console.log(
				`WebSocket server exited with code ${code}, signal ${signal}`
			);
			if (code !== 0) {
				console.error(`WebSocket server crashed. Exit code: ${code}`);
			}
			serverProcess = null;
		});

		// Start Next.js server
		const nextServer = http.createServer((req, res) => handle(req, res));
		const port = 3000;
		nextServer.listen(port, async () => {
			console.log(`Next.js running on http://localhost:${port}`);
			try {
				await waitOn({
					resources: [
						`http://localhost:${port}`, // Next.js
						`tcp:localhost:4000`, // WebSocket server
						// Add `http://localhost:5000` if Python backend exposes an API
					],
					timeout: 30000, // 30 seconds timeout
				});
				console.log("All services are ready");
				win.loadURL(`http://localhost:${port}`);
				if (dev) win.webContents.openDevTools();
			} catch (error) {
				console.error("Failed to verify services:", error);
				app.quit();
			}
		});

		win.on("closed", async () => {
			console.log("Window closed, cleaning up...");
			if (pythonProcess) {
				console.log("Forcefully killing Python process...");
				terminateProcess(pythonProcess, async () => {
					await waitForProcessTermination(pythonProcess);
					pythonProcess = null;
				});
			}
			if (serverProcess) {
				console.log("Forcefully killing WebSocket server process...");
				terminateProcess(serverProcess, async () => {
					await waitForProcessTermination(serverProcess);
					serverProcess = null;
				});
			}
			win = null;
		});
	} catch (error) {
		console.error("Error in createWindow:", error);
		app.quit();
	}
}

app.whenReady().then(async () => {
	try {
		await createWindow();
	} catch (error) {
		console.error("Failed to start Electron app:", error);
		app.quit();
	}
});

app.on("window-all-closed", async () => {
	console.log("All windows closed, quitting app...");
	if (process.platform !== "darwin") {
		if (pythonProcess) {
			console.log("Forcefully killing Python process on app quit...");
			terminateProcess(pythonProcess, async () => {
				await waitForProcessTermination(pythonProcess);
				pythonProcess = null;
				if (!serverProcess) app.quit();
			});
		}
		if (serverProcess) {
			console.log("Forcefully killing WebSocket server process on app quit...");
			terminateProcess(serverProcess, async () => {
				await waitForProcessTermination(serverProcess);
				serverProcess = null;
				if (!pythonProcess) app.quit();
			});
		}
		if (!pythonProcess && !serverProcess) {
			app.quit();
		}
	}
});

app.on("activate", async () => {
	if (win === null) await createWindow();
});

// Ensure child processes are killed on app exit
app.on("before-quit", async () => {
	console.log("App is quitting, ensuring all processes are terminated...");
	if (pythonProcess) {
		terminateProcess(pythonProcess, async () => {
			await waitForProcessTermination(pythonProcess);
			pythonProcess = null;
		});
	}
	if (serverProcess) {
		terminateProcess(serverProcess, async () => {
			await waitForProcessTermination(serverProcess);
			serverProcess = null;
		});
	}
});
