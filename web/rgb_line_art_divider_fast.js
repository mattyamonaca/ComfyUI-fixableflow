/**
 * RGBLineArtDividerFast Web Extension
 * Adds a download button to the node for PSD file download
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "ComfyUI-fixableflow.RGBLineArtDividerFast",
    
    async nodeCreated(node) {
        // Only apply to RGBLineArtDividerFast nodes
        if (node.comfyClass !== "RGBLineArtDividerFast") {
            return;
        }
        
        // Store the PSD path when the node executes
        node.psdPath = null;
        
        // Add custom widget for download button
        const downloadWidget = node.addWidget(
            "button",
            "Download PSD",
            null,
            () => {
                if (node.psdPath) {
                    // Extract just the filename from the full path
                    const filename = node.psdPath.split('/').pop() || node.psdPath.split('\\').pop();
                    console.log("Downloading file:", filename, "from path:", node.psdPath);
                    
                    // ComfyUI standard /view endpoint format
                    // type=output means the file is in the output directory
                    const params = new URLSearchParams({
                        filename: filename,
                        type: 'output'
                    });
                    const downloadUrl = `/view?${params.toString()}`;
                    
                    // Create a temporary link and click it
                    const link = document.createElement('a');
                    link.href = downloadUrl;
                    link.download = filename;
                    link.style.display = 'none';
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                    
                    // Show success message with shortened timeout
                    if (app.ui.dialog) {
                        app.ui.dialog.show(`Downloading: ${filename}`);
                        setTimeout(() => {
                            if (app.ui.dialog && app.ui.dialog.close) {
                                app.ui.dialog.close();
                            }
                        }, 2000);
                    }
                    console.log("Download initiated for:", filename);
                } else {
                    console.warn("No PSD file path available");
                    if (app.ui.dialog) {
                        app.ui.dialog.show("No PSD file available. Please run the workflow first.");
                    }
                }
            },
            { 
                serialize: false 
            }
        );
        
        // Style the button
        downloadWidget.disabled = true;
        downloadWidget.bgColor = "#444444";
        
        // Override the onExecuted method to capture the PSD path
        const originalOnExecuted = node.onExecuted;
        node.onExecuted = function(output) {
            // Debug logging
            console.log("RGBLineArtDividerFast onExecuted called with:", output);
            
            // Call original handler if it exists
            if (originalOnExecuted) {
                originalOnExecuted.call(this, output);
            }
            
            // Try different ways to extract the PSD path
            let psdPath = null;
            
            // Method 1: Direct array access (ComfyUI typically returns arrays)
            if (output && Array.isArray(output)) {
                // The 4th element (index 3) should be the psd_path
                if (output.length > 3 && output[3]) {
                    psdPath = Array.isArray(output[3]) ? output[3][0] : output[3];
                    console.log("Method 1 - Found PSD path from array:", psdPath);
                }
            }
            
            // Method 2: Named property access
            if (!psdPath && output && output.psd_path) {
                psdPath = Array.isArray(output.psd_path) ? output.psd_path[0] : output.psd_path;
                console.log("Method 2 - Found PSD path from property:", psdPath);
            }
            
            // Method 3: Check if output has a nested structure
            if (!psdPath && output && output.output) {
                if (Array.isArray(output.output) && output.output.length > 3) {
                    psdPath = Array.isArray(output.output[3]) ? output.output[3][0] : output.output[3];
                    console.log("Method 3 - Found PSD path from nested structure:", psdPath);
                }
            }
            
            // Update button if we found a path
            if (psdPath) {
                node.psdPath = psdPath;
                downloadWidget.disabled = false;
                downloadWidget.bgColor = "#4CAF50";
                const fileName = psdPath.split('/').pop() || psdPath.split('\\').pop() || 'output.psd';
                downloadWidget.name = `Download: ${fileName}`;
                console.log("PSD download enabled for:", psdPath);
            } else {
                console.warn("Could not find PSD path in output");
            }
        };
        
        // Reset button when node is cleared
        const originalOnClear = node.onClear;
        node.onClear = function() {
            if (originalOnClear) {
                originalOnClear.call(this);
            }
            node.psdPath = null;
            downloadWidget.disabled = true;
            downloadWidget.bgColor = "#444444";
            downloadWidget.name = "Download PSD";
        };
    },
});
