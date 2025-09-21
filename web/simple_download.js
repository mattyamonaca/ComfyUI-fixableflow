/**
 * RGBLineArtDividerFast - Simple Web Extension
 * Adds a simple manual download button to the node
 */

import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "ComfyUI-fixableflow.RGBLineArtDividerFastSimple",
    
    async nodeCreated(node) {
        // Only apply to RGBLineArtDividerFast nodes
        if (node.comfyClass === "RGBLineArtDividerFast") {
            console.log("Adding simple download button to RGBLineArtDividerFast node");
            
            // Get the 4th widget (psd_path output) value when available
            const getOutputValue = () => {
                // Check if node has been executed and has outputs
                if (node.widgets_values && node.widgets_values.length > 3) {
                    return node.widgets_values[3];
                }
                
                // Alternative: check the connected widgets
                if (node.widgets) {
                    for (let widget of node.widgets) {
                        if (widget.name === "psd_path" && widget.value) {
                            return widget.value;
                        }
                    }
                }
                
                return null;
            };
            
            // Add a simple button that checks for the filename
            const button = node.addWidget(
                "button",
                "Download PSD (Manual)",
                "Download PSD",
                () => {
                    console.log("Manual download button clicked");
                    
                    // Try to get filename from output value
                    let filename = getOutputValue();
                    
                    if (!filename) {
                        // Prompt user to enter filename if not found
                        filename = prompt("Enter the PSD filename (e.g., output_rgb_fast_normal_abc123.psd):");
                    }
                    
                    if (filename) {
                        // Clean up the filename (remove path if present)
                        filename = filename.split('/').pop() || filename.split('\\').pop() || filename;
                        
                        console.log("Downloading:", filename);
                        
                        // Create download URL
                        const downloadUrl = `/view?filename=${encodeURIComponent(filename)}&type=output`;
                        
                        // Create and click download link
                        const link = document.createElement('a');
                        link.href = downloadUrl;
                        link.download = filename;
                        link.style.display = 'none';
                        document.body.appendChild(link);
                        link.click();
                        document.body.removeChild(link);
                        
                        console.log("Download initiated for:", filename);
                    }
                }
            );
            
            // Store button reference
            node.downloadButton = button;
            
            console.log("Simple download button added to node");
        }
    }
});
