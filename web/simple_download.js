/**
 * RGBLineArtDividerFast Web Extension
 * Log file based download system
 */

import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "ComfyUI-fixableflow.RGBLineArtDividerFast",
    
    async nodeCreated(node) {
        // Only apply to RGBLineArtDividerFast nodes  
        if (node.comfyClass === "RGBLineArtDividerFast") {
            console.log("[RGBDivider] Setting up download button");
            
            // Store the latest generated filename
            let latestPsdFilename = null;
            let checkInterval = null;
            
            // Add download button
            const downloadButton = node.addWidget(
                "button",
                "Download PSD",
                "⬇ Download PSD (Run workflow first)",
                async () => {
                    // First check if we have a cached filename
                    if (latestPsdFilename) {
                        downloadPsd(latestPsdFilename);
                        return;
                    }
                    
                    // Try to read the log file
                    try {
                        const logFilename = await fetchLatestPsdFromLog();
                        if (logFilename) {
                            latestPsdFilename = logFilename;
                            updateButton(logFilename);
                            downloadPsd(logFilename);
                        } else {
                            // Manual fallback
                            promptForManualInput();
                        }
                    } catch (error) {
                        console.error("[RGBDivider] Error reading log:", error);
                        promptForManualInput();
                    }
                }
            );
            
            // Function to fetch latest PSD filename from log
            async function fetchLatestPsdFromLog() {
                try {
                    // Try to fetch the log file
                    const response = await fetch('/view?filename=fixableflow_savepath.log&type=output&t=' + Date.now());
                    
                    if (response.ok) {
                        const text = await response.text();
                        const filename = text.trim();
                        
                        if (filename && filename.includes('.psd')) {
                            console.log("[RGBDivider] Found filename in log:", filename);
                            return filename;
                        }
                    }
                } catch (error) {
                    console.error("[RGBDivider] Failed to read log file:", error);
                }
                return null;
            }
            
            // Function to download PSD
            function downloadPsd(filename) {
                console.log("[RGBDivider] Downloading:", filename);
                
                // Ensure we have just the filename, not the full path
                const cleanFilename = filename.split('/').pop() || filename.split('\\').pop() || filename;
                
                const downloadUrl = `/view?filename=${encodeURIComponent(cleanFilename)}&type=output`;
                console.log("[RGBDivider] Download URL:", downloadUrl);
                
                const link = document.createElement('a');
                link.href = downloadUrl;
                link.download = cleanFilename;
                link.style.display = 'none';
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                
                console.log("[RGBDivider] Download initiated");
            }
            
            // Function for manual input
            function promptForManualInput() {
                const filename = prompt(
                    "ログファイルが見つかりません。\n" +
                    "サーバーコンソールからファイル名をコピーしてください。\n" +
                    "例: output_rgb_fast_normal_G3HTgU8UPn.psd"
                );
                
                if (filename && filename.includes('.psd')) {
                    const cleanFilename = filename.split('/').pop() || filename.split('\\').pop() || filename;
                    latestPsdFilename = cleanFilename;
                    updateButton(cleanFilename);
                    downloadPsd(cleanFilename);
                }
            }
            
            // Helper function to update button
            function updateButton(filename) {
                const cleanFilename = filename.split('/').pop() || filename.split('\\').pop() || filename;
                downloadButton.name = `⬇ Download: ${cleanFilename}`;
                downloadButton.color = "#4CAF50";
                downloadButton.bgcolor = "#2E7D32";
                console.log("[RGBDivider] Button updated:", cleanFilename);
            }
            
            // Initial button style
            downloadButton.color = "#888888";
            downloadButton.bgcolor = "#333333";
            
            // Start periodic check for log file updates
            function startLogMonitoring() {
                if (checkInterval) {
                    clearInterval(checkInterval);
                }
                
                // Check log file every 2 seconds after workflow execution
                checkInterval = setInterval(async () => {
                    const filename = await fetchLatestPsdFromLog();
                    if (filename && filename !== latestPsdFilename) {
                        console.log("[RGBDivider] New PSD detected:", filename);
                        latestPsdFilename = filename;
                        updateButton(filename);
                        
                        // Stop checking once we found a file
                        clearInterval(checkInterval);
                        checkInterval = null;
                    }
                }, 2000);
                
                // Stop checking after 10 seconds to avoid unnecessary requests
                setTimeout(() => {
                    if (checkInterval) {
                        clearInterval(checkInterval);
                        checkInterval = null;
                        console.log("[RGBDivider] Stopped log monitoring");
                    }
                }, 10000);
            }
            
            // Monitor node execution to trigger log checking
            const originalOnExecuted = node.onExecuted;
            node.onExecuted = function(output) {
                console.log("[RGBDivider] Node executed");
                
                if (originalOnExecuted) {
                    originalOnExecuted.apply(this, arguments);
                }
                
                // Start monitoring log file after execution
                console.log("[RGBDivider] Starting log file monitoring...");
                startLogMonitoring();
            };
            
            // Also check log immediately on button creation in case file already exists
            (async () => {
                const filename = await fetchLatestPsdFromLog();
                if (filename) {
                    latestPsdFilename = filename;
                    updateButton(filename);
                    console.log("[RGBDivider] Found existing PSD:", filename);
                }
            })();
            
            console.log("[RGBDivider] Download button ready");
        }
    }
});

// Global helper function for debugging
window.checkPsdLog = async function() {
    try {
        const response = await fetch('/view?filename=fixableflow_savepath.log&type=output&t=' + Date.now());
        if (response.ok) {
            const text = await response.text();
            console.log("Log file content:", text);
            return text;
        } else {
            console.log("Log file not found or not accessible");
        }
    } catch (error) {
        console.error("Error reading log:", error);
    }
    return null;
};

console.log("[RGBLineArtDividerFast] Extension loaded - log file mode");
