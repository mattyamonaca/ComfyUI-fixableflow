import { app } from "../../scripts/app.js";

// テスト用のシンプルな実装
console.log("=== RGB Line Art Divider Fast Extension Loading ===");

app.registerExtension({
    name: "ComfyUI.RGBLineArtDividerFastTest",
    
    async init() {
        console.log("RGB Line Art Divider Fast Extension initialized");
    },
    
    async setup() {
        console.log("RGB Line Art Divider Fast Extension setup");
    },
    
    async nodeCreated(node) {
        // ノードタイプを確認
        console.log("Node created:", node.comfyClass);
        
        if (node.comfyClass === "RGBLineArtDividerFast") {
            console.log("✅ Found RGBLineArtDividerFast node!");
            
            // ボタンを追加
            try {
                const widget = node.addWidget(
                    "button",
                    "Test Download",
                    "Click to Test",
                    function() {
                        console.log("Button clicked!");
                        alert("Download button clicked! Check console for details.");
                    }
                );
                console.log("✅ Button added successfully");
            } catch (error) {
                console.error("❌ Failed to add button:", error);
            }
        }
    }
});

console.log("=== RGB Line Art Divider Fast Extension Loaded ===");
