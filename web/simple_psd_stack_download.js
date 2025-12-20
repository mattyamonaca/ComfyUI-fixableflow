/**
 * SimplePSDStackNode Web Extension
 * Download button for Simple PSD Stack
 */

import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "ComfyUI-fixableflow.SimplePSDStack",
    
    async nodeCreated(node) {
        // Only apply to SimplePSDStackNode nodes  
        if (node.comfyClass === "SimplePSDStackNode") {
            console.log("[SimplePSDStack] Setting up download button");
            
            // Add download button
            const downloadButton = node.addWidget(
                "button",
                "Download PSD",
                "⬇ Download Latest PSD",
                async () => {
                    console.log("[SimplePSDStack] Download button clicked");
                    
                    // Always fetch the latest filename from log file
                    try {
                        // Fetch with cache-busting timestamp
                        const response = await fetch('/view?filename=simple_psd_stack_savepath.log&type=output&t=' + Date.now());
                        
                        if (response.ok) {
                            const text = await response.text();
                            const filename = text.trim();
                            
                            if (filename && filename.includes('.psd')) {
                                console.log("[SimplePSDStack] Found file in log:", filename);
                                
                                // Download immediately
                                const cleanFilename = filename.split('/').pop() || filename.split('\\\\').pop() || filename;
                                const downloadUrl = `/view?filename=${encodeURIComponent(cleanFilename)}&type=output`;
                                console.log("[SimplePSDStack] Download URL:", downloadUrl);
                                
                                const link = document.createElement('a');
                                link.href = downloadUrl;
                                link.download = cleanFilename;
                                link.style.display = 'none';
                                document.body.appendChild(link);
                                link.click();
                                document.body.removeChild(link);
                                
                                console.log("[SimplePSDStack] Downloaded:", cleanFilename);
                            } else {
                                console.log("[SimplePSDStack] Log file exists but no valid PSD filename found");
                                promptForManualInput();
                            }
                        } else {
                            console.log("[SimplePSDStack] Log file not found (404)");
                            promptForManualInput();
                        }
                    } catch (error) {
                        console.error("[SimplePSDStack] Error reading log:", error);
                        promptForManualInput();
                    }
                    
                    // Function for manual input
                    function promptForManualInput() {
                        const filename = prompt(
                            "ログファイルが見つかりません。\\n" +
                            "ワークフローを実行するか、手動でファイル名を入力してください。\\n" +
                            "例: layered_20251220_153045.psd"
                        );
                        
                        if (filename && filename.includes('.psd')) {
                            const cleanFilename = filename.split('/').pop() || filename.split('\\\\').pop() || filename;
                            console.log("[SimplePSDStack] Manual input:", cleanFilename);
                            
                            const downloadUrl = `/view?filename=${encodeURIComponent(cleanFilename)}&type=output`;
                            const link = document.createElement('a');
                            link.href = downloadUrl;
                            link.download = cleanFilename;
                            document.body.appendChild(link);
                            link.click();
                            document.body.removeChild(link);
                            
                            console.log("[SimplePSDStack] Manual download completed");
                        }
                    }
                }
            );
            
            // Simple button style
            downloadButton.color = "#10B981";
            downloadButton.bgcolor = "#059669";
            
            console.log("[SimplePSDStack] Download button ready");
        }
    }
});

console.log("[SimplePSDStack] Extension loaded");
