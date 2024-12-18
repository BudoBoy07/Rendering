let canvas, context, device, swapChain;

async function init() {
    canvas = document.getElementById('myCanvas');
    context = canvas.getContext('webgpu');

    if (!context) {
        console.error('WebGPU not supported');
        return;
    }

    device = await navigator.gpu.requestAdapter().then(adapter => adapter.requestDevice());

    const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
        device: device,
        format: canvasFormat,
    });

    render();
}

function render() {
    const commandEncoder = device.createCommandEncoder();
    const textureView = context.getCurrentTexture().createView();

    const renderPassDescriptor = {
        colorAttachments: [{
            view: textureView,
            clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
            loadOp: 'clear',
            storeOp: 'store',
        }],
    };

    const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
    passEncoder.end();

    device.queue.submit([commandEncoder.finish()]);
}

window.addEventListener('load', init);