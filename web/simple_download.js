/**
 * RGBLineArtDividerFast Web Extension
 * Smart auto-download - generates filename based on node execution
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "ComfyUI-fixableflow.RGBLineArtDividerFast",
    
    async nodeCreated(node) {
        // Only apply to RGBLineArtDividerFast nodes  
        if (node.comfyClass === "RGBLineArtDividerFast") {
            console.log("RGBLineArtDividerFast: Setting up smart download");
            
            // Store the node's generated filename
            let nodeFilename = null;
            let executionCounter = 0;
            
            // Add download button
            const downloadButton = node.addWidget(
                "button",
                "Download PSD",
                "⬇ Download PSD (Run workflow first)",
                () => {
                    if (nodeFilename) {
                        console.log("Downloading:", nodeFilename);
                        
                        // Create download URL using the known path structure
                        const downloadUrl = `/view?filename=${encodeURIComponent(nodeFilename)}&type=output`;
                        
                        // Create and click download link
                        const link = document.createElement('a');
                        link.href = downloadUrl;
                        link.download = nodeFilename;
                        link.style.display = 'none';
                        document.body.appendChild(link);
                        link.click();
                        document.body.removeChild(link);
                        
                        console.log("Download completed for:", nodeFilename);
                    } else {
                        alert("No PSD file available. Please run the workflow first.");
                    }
                }
            );
            
            // Initial style - disabled appearance
            downloadButton.color = "#888888";
            downloadButton.bgcolor = "#333333";
            
            // Monitor node execution
            const originalOnExecuted = node.onExecuted;
            node.onExecuted = function(output) {
                console.log("Node executed with output:", output);
                
                if (originalOnExecuted) {
                    originalOnExecuted.apply(this, arguments);
                }
                
                // Increment execution counter
                executionCounter++;
                
                // Try to extract filename from output
                let capturedFilename = null;
                
                if (output) {
                    // Check if it's an array (ComfyUI standard output format)
                    if (Array.isArray(output)) {
                        // Check for 4th element (psd_filename)
                        if (output.length > 3 && output[3]) {
                            // Handle both string and array formats
                            if (typeof output[3] === 'string') {
                                capturedFilename = output[3];
                            } else if (Array.isArray(output[3]) && output[3][0]) {
                                capturedFilename = output[3][0];
                            }
                        }
                    }
                    // Check for object format
                    else if (typeof output === 'object') {
                        if (output.psd_filename) {
                            capturedFilename = Array.isArray(output.psd_filename) ? 
                                output.psd_filename[0] : output.psd_filename;
                        } else if (output.text && Array.isArray(output.text)) {
                            // Sometimes string outputs are in text array
                            const lastText = output.text[output.text.length - 1];
                            if (lastText && lastText.includes('.psd')) {
                                capturedFilename = lastText;
                            }
                        }
                    }
                }
                
                if (capturedFilename && capturedFilename.includes('.psd')) {
                    nodeFilename = capturedFilename;
                    downloadButton.name = `⬇ Download: ${capturedFilename}`;
                    downloadButton.color = "#4CAF50";
                    downloadButton.bgcolor = "#2E7D32";
                    console.log("Captured filename from output:", capturedFilename);
                } else {
                    console.log("No filename in output, will wait for other sources");
                }
            };
            
            // Store node reference for global access
            node.rgbDividerNode = true;
            node.updateFilename = function(filename) {
                if (filename && filename.includes('.psd')) {
                    nodeFilename = filename;
                    downloadButton.name = `⬇ Download: ${filename}`;
                    downloadButton.color = "#4CAF50";
                    downloadButton.bgcolor = "#2E7D32";
                    console.log("Updated filename:", filename);
                }
            };
            
            console.log("RGBLineArtDividerFast: Smart download ready");
        }
    }
});

// Monitor WebSocket messages for PSD filenames
if (api.socket) {
    const ws = api.socket;
    const originalOnMessage = ws.onmessage;
    
    ws.onmessage = function(event) {
        try {
            const msg = JSON.parse(event.data);
            
            // Check for execution complete messages
            if (msg.type === 'executed' && msg.data && msg.data.output) {
                // Look for psd_filename in the output
                if (msg.data.output.psd_filename) {
                    const filename = Array.isArray(msg.data.output.psd_filename) ? 
                        msg.data.output.psd_filename[0] : msg.data.output.psd_filename;
                    
                    if (filename && filename.includes('.psd')) {
                        console.log("Found PSD filename in WebSocket:", filename);
                        
                        // Update all RGBLineArtDividerFast nodes
                        if (app.graph && app.graph.nodes) {
                            app.graph.nodes.forEach(node => {
                                if (node.rgbDividerNode && node.updateFilename) {
                                    node.updateFilename(filename);
                                }
                            });
                        }
                    }
                }
            }
            
            // Also check console messages for PSD file saved
            if (msg.type === 'console' && msg.data) {
                const text = typeof msg.data === 'string' ? msg.data : JSON.stringify(msg.data);
                if (text.includes('[RGBLineArtDividerFast]') && text.includes('PSD file saved:')) {
                    // Extract filename from the log
                    const match = text.match(/output_rgb_fast_normal_[A-Za-z0-9]+\.psd/);
                    if (match) {
                        const filename = match[0];
                        console.log("Found PSD filename in console:", filename);
                        
                        // Update all RGBLineArtDividerFast nodes
                        if (app.graph && app.graph.nodes) {
                            app.graph.nodes.forEach(node => {
                                if (node.rgbDividerNode && node.updateFilename) {
                                    node.updateFilename(filename);
                                }
                            });
                        }
                    }
                }
            }
        } catch (e) {
            // Ignore parse errors
        }
        
        if (originalOnMessage) {
            originalOnMessage.apply(this, arguments);
        }
    };
}

// Monitor API events
api.addEventListener("executed", (event) => {
    if (event.detail && event.detail.output) {
        // Check for psd_filename in the output
        if (event.detail.output.psd_filename) {
            const filename = Array.isArray(event.detail.output.psd_filename) ? 
                event.detail.output.psd_filename[0] : event.detail.output.psd_filename;
            
            if (filename && filename.includes('.psd')) {
                console.log("Found PSD filename in API event:", filename);
                
                // Update all RGBLineArtDividerFast nodes
                if (app.graph && app.graph.nodes) {
                    app.graph.nodes.forEach(node => {
                        if (node.rgbDividerNode && node.updateFilename) {
                            node.updateFilename(filename);
                        }
                    });
                }
            }
        }
    }
});

console.log("RGBLineArtDividerFast extension loaded - smart download enabled");
