/**
 * RGBLineArtDividerFast Web Extension - Alternative Implementation
 * Adds a download button to the node for PSD file download
 */

import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

app.registerExtension({
    name: "ComfyUI-fixableflow.RGBDividerFast",
    
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "RGBLineArtDividerFast") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated?.apply(this, arguments);
                
                const node = this;
                console.log("RGBLineArtDividerFast node created:", node);
                
                // Add download button widget
                const downloadButton = this.addWidget(
                    "button",
                    "Download PSD",
                    "Download PSD",
                    function() {
                        console.log("Download button clicked");
                        
                        // Get the last execution output
                        const outputs = node.outputs;
                        if (outputs && outputs.length > 3) {
                            // The 4th output should be the PSD path
                            const psdPathOutput = outputs[3];
                            console.log("PSD path output:", psdPathOutput);
                            
                            if (psdPathOutput && psdPathOutput.links && psdPathOutput.links.length > 0) {
                                // Get the actual value from the link
                                const linkId = psdPathOutput.links[0];
                                const link = app.graph.links[linkId];
                                if (link) {
                                    const originNode = app.graph.getNodeById(link.origin_id);
                                    if (originNode && originNode.widgets_values) {
                                        const psdPath = originNode.widgets_values[link.origin_slot];
                                        console.log("Found PSD path:", psdPath);
                                        
                                        if (psdPath) {
                                            const filename = psdPath.split('/').pop() || psdPath.split('\\').pop();
                                            const downloadUrl = `/view?filename=${encodeURIComponent(filename)}&type=output`;
                                            
                                            const a = document.createElement('a');
                                            a.href = downloadUrl;
                                            a.download = filename;
                                            document.body.appendChild(a);
                                            a.click();
                                            document.body.removeChild(a);
                                            
                                            console.log("Download initiated for:", filename);
                                        }
                                    }
                                }
                            }
                        }
                        
                        // Alternative method: Check node properties
                        if (node.properties && node.properties.psd_path) {
                            const psdPath = node.properties.psd_path;
                            const filename = psdPath.split('/').pop() || psdPath.split('\\').pop();
                            const downloadUrl = `/view?filename=${encodeURIComponent(filename)}&type=output`;
                            
                            const a = document.createElement('a');
                            a.href = downloadUrl;
                            a.download = filename;
                            document.body.appendChild(a);
                            a.click();
                            document.body.removeChild(a);
                            
                            console.log("Download initiated for:", filename);
                        } else {
                            console.warn("No PSD file path available. Please run the workflow first.");
                            alert("No PSD file available. Please run the workflow first.");
                        }
                    }
                );
                
                // Store reference to the button
                this.downloadButton = downloadButton;
                
                // Override onExecuted to update button state
                const originalOnExecuted = this.onExecuted;
                this.onExecuted = function(message) {
                    console.log("Node executed with message:", message);
                    
                    if (originalOnExecuted) {
                        originalOnExecuted.apply(this, arguments);
                    }
                    
                    // Try to extract PSD path from various sources
                    let psdPath = null;
                    
                    // Check if message contains the path
                    if (message) {
                        console.log("Message type:", typeof message);
                        console.log("Message keys:", Object.keys(message));
                        
                        // Try different extraction methods
                        if (message.psd_path) {
                            psdPath = Array.isArray(message.psd_path) ? message.psd_path[0] : message.psd_path;
                        } else if (message.text && Array.isArray(message.text) && message.text.length > 0) {
                            // Sometimes string outputs come as text array
                            const lastText = message.text[message.text.length - 1];
                            if (lastText && lastText.includes('.psd')) {
                                psdPath = lastText;
                            }
                        } else if (Array.isArray(message) && message.length > 3) {
                            // Direct array format
                            psdPath = message[3];
                        }
                    }
                    
                    // Store the path in node properties
                    if (psdPath) {
                        console.log("Storing PSD path:", psdPath);
                        this.properties = this.properties || {};
                        this.properties.psd_path = psdPath;
                        
                        // Update button appearance
                        if (this.downloadButton) {
                            this.downloadButton.name = `Download: ${psdPath.split('/').pop() || 'PSD'}`;
                        }
                    }
                };
                
                return result;
            };
        }
    }
});
