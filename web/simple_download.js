/**
 * RGBLineArtDividerFast Web Extension
 * Pure stateless - no caching at all
 */

import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "ComfyUI-fixableflow.RGBLineArtDividerFast",
    
    async nodeCreated(node) {
        // Only apply to RGBLineArtDividerFast nodes  
        if (node.comfyClass === "RGBLineArtDividerFast") {
            console.log("[RGBDivider] Setting up download button");
            
            // Add download button - simple and stateless
            const downloadButton = node.addWidget(
                "button",
                "Download PSD",
                "⬇ Download Latest PSD",
                async () => {
                    console.log("[RGBDivider] Download button clicked");
                    
                    // Always fetch the latest filename from log file
                    try {
                        // Fetch with cache-busting timestamp
                        const response = await fetch('/view?filename=fixableflow_savepath.log&type=output&t=' + Date.now());
                        
                        if (response.ok) {
                            const text = await response.text();
                            const filename = text.trim();
                            
                            if (filename && filename.includes('.psd')) {
                                console.log("[RGBDivider] Found file in log:", filename);
                                
                                // Download immediately
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
                                
                                console.log("[RGBDivider] Downloaded:", cleanFilename);
                            } else {
                                console.log("[RGBDivider] Log file exists but no valid PSD filename found");
                                promptForManualInput();
                            }
                        } else {
                            console.log("[RGBDivider] Log file not found (404)");
                            promptForManualInput();
                        }
                    } catch (error) {
                        console.error("[RGBDivider] Error reading log:", error);
                        promptForManualInput();
                    }
                    
                    // Function for manual input (defined inline to avoid any state)
                    function promptForManualInput() {
                        const filename = prompt(
                            "ログファイルが見つかりません。\n" +
                            "ワークフローを実行するか、手動でファイル名を入力してください。\n" +
                            "例: output_rgb_fast_normal_G3HTgU8UPn.psd"
                        );
                        
                        if (filename && filename.includes('.psd')) {
                            const cleanFilename = filename.split('/').pop() || filename.split('\\').pop() || filename;
                            console.log("[RGBDivider] Manual input:", cleanFilename);
                            
                            const downloadUrl = `/view?filename=${encodeURIComponent(cleanFilename)}&type=output`;
                            const link = document.createElement('a');
                            link.href = downloadUrl;
                            link.download = cleanFilename;
                            document.body.appendChild(link);
                            link.click();
                            document.body.removeChild(link);
                            
                            console.log("[RGBDivider] Manual download completed");
                        }
                    }
                }
            );
            
            // Simple button style
            downloadButton.color = "#3B82F6";
            downloadButton.bgcolor = "#1E40AF";
            
            console.log("[RGBDivider] Download button ready");
        }
    }
});

// Global helper function for debugging (stateless)
window.checkPsdLog = async function() {
    try {
        const response = await fetch('/view?filename=fixableflow_savepath.log&type=output&t=' + Date.now());
        if (response.ok) {
            const text = await response.text();
            console.log("Log file content:", text);
            return text.trim();
        } else {
            console.log("Log file not found (404)");
        }
    } catch (error) {
        console.error("Error reading log:", error);
    }
    return null;
};

// Global function to manually trigger download (stateless)
window.downloadLatestPsd = async function() {
    const response = await fetch('/view?filename=fixableflow_savepath.log&type=output&t=' + Date.now());
    if (response.ok) {
        const filename = (await response.text()).trim();
        if (filename && filename.includes('.psd')) {
            const cleanFilename = filename.split('/').pop() || filename.split('\\').pop() || filename;
            const downloadUrl = `/view?filename=${encodeURIComponent(cleanFilename)}&type=output`;
            
            const link = document.createElement('a');
            link.href = downloadUrl;
            link.download = cleanFilename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            
            console.log("Downloaded:", cleanFilename);
            return true;
        }
    }
    console.log("No PSD file found in log");
    return false;
};

console.log("[RGBLineArtDividerFast] Extension loaded - pure stateless mode");
