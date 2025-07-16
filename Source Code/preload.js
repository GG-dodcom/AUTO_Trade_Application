const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("electronAPI", {
	sendToPython: (data) => ipcRenderer.invoke("python-request", data),
	onPythonResponse: (callback) => ipcRenderer.on("python-response", callback),
});
