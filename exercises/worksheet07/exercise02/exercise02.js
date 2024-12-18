async function init() {
    if (!navigator.gpu) {
        alert("WebGPU not supported on this browser.");
        return;
    }


    const canvas = document.getElementById("myCanvas");
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();

    const context = canvas.getContext("webgpu");
    const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
        device: device,
        format: canvasFormat,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST,
    });


    const cornellBoxData = await readOBJFile('CornellBoxWithBlocks.obj', 1.0, false);



    const vertexBuffer = device.createBuffer({
        size: cornellBoxData.vertices.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(vertexBuffer, 0, cornellBoxData.vertices);

    const indexBuffer = device.createBuffer({
        size: cornellBoxData.indices.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(indexBuffer, 0, cornellBoxData.indices);

    const normalBuffer = device.createBuffer({
        size: cornellBoxData.normals.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(normalBuffer, 0, cornellBoxData.normals);

    const materialBuffer = device.createBuffer({
        size: cornellBoxData.materials.length * 8 * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const materialIndexBuffer = device.createBuffer({
        size: cornellBoxData.mat_indices.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(materialIndexBuffer, 0, cornellBoxData.mat_indices);

    const lightIndicesBuffer = device.createBuffer({
        size: cornellBoxData.light_indices.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(lightIndicesBuffer, 0, cornellBoxData.light_indices);



    const materialData = new Float32Array(cornellBoxData.materials.length * 8);
    cornellBoxData.materials.forEach((material, index) => {

        materialData[index * 8 + 0] = material.color.r || 0;
        materialData[index * 8 + 1] = material.color.g || 0;
        materialData[index * 8 + 2] = material.color.b || 0;
        materialData[index * 8 + 3] = material.color.a || 1.0;


        materialData[index * 8 + 4] = material.emission ? material.emission.r : 0;
        materialData[index * 8 + 5] = material.emission ? material.emission.g : 0;
        materialData[index * 8 + 6] = material.emission ? material.emission.b : 0;
        materialData[index * 8 + 7] = material.emission ? material.emission.a || 0 : 0;
    });
    device.queue.writeBuffer(materialBuffer, 0, materialData);

    const shader = `
        struct Uniforms {
            aspectRatio: f32,
            frameCount: f32,
            width: f32,
            height: f32,
        }
        @group(0) @binding(0) var<uniform> uniforms: Uniforms;
        @group(0) @binding(1) var<storage, read> vertices: array<vec4<f32>>;
        @group(0) @binding(2) var<storage, read> indices: array<vec4<u32>>;
        @group(0) @binding(3) var<storage, read> normals: array<vec4<f32>>;
        @group(0) @binding(4) var<storage, read> materials: array<vec4<f32>>;
        @group(0) @binding(5) var<storage, read> materialIndices: array<u32>;
        @group(0) @binding(6) var<storage, read> lightIndices: array<u32>;
        @group(0) @binding(7) var previousFrame: texture_2d<f32>;

        fn rand(seed: vec2<f32>) -> f32 {
            return fract(sin(dot(seed, vec2<f32>(12.9898, 78.233))) * 43758.5453);
        }

        fn getRandom2D(seed: vec2<f32>) -> vec2<f32> {
            return vec2<f32>(
                rand(seed),
                rand(seed + vec2<f32>(1.0, 1.0))
            );
        }

        struct VertexOutput {
            @builtin(position) position: vec4<f32>,
            @location(0) uv: vec2<f32>,
        }

        struct Ray {
            origin: vec3<f32>,
            direction: vec3<f32>,
            tmin: f32,
            tmax: f32,
        }

        struct HitInfo {
            hit: bool,
            distance: f32,
            position: vec3<f32>,
            normal: vec3<f32>,
            materialIndex: u32,
        }

        @vertex
        fn vertexMain(@builtin(vertex_index) vertexIndex : u32) -> VertexOutput {
            var pos = array<vec2<f32>, 6>(
                vec2<f32>(-1.0, -1.0),
                vec2<f32>(1.0, -1.0),
                vec2<f32>(-1.0, 1.0),
                vec2<f32>(-1.0, 1.0),
                vec2<f32>(1.0, -1.0),
                vec2<f32>(1.0, 1.0)
            );
            var output: VertexOutput;
            output.position = vec4<f32>(pos[vertexIndex], 0.0, 1.0);
            output.uv = pos[vertexIndex];
            return output;
        }

        fn generateRay(uv: vec2<f32>) -> Ray {
            let eye = vec3<f32>(277.0, 275.0, -570.0);
            let lookAt = vec3<f32>(277.0, 275.0, 0.0);
            let up = vec3<f32>(0.0, -1.0, 0.0);

            let w = normalize(eye - lookAt);
            let u = normalize(cross(up, w));
            let v = cross(w, u);

            let aspect = uniforms.aspectRatio;
            let d = 1.0;

            let x = uv.x * aspect;
            let y = uv.y;

            var ray: Ray;
            ray.origin = eye;
            ray.direction = normalize(x*u + y*v - d*w);
            ray.tmin = 0.001;
            ray.tmax = 1000.0;

            return ray;
        }

        fn intersectTriangle(ray: Ray, v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>, n0: vec3<f32>, n1: vec3<f32>, n2: vec3<f32>, materialIndex: u32) -> HitInfo {
            var hit: HitInfo;
            hit.hit = false;

            let e1 = v1 - v0;
            let e2 = v2 - v0;
            let h = cross(ray.direction, e2);
            let a = dot(e1, h);

            if (abs(a) < 0.0001) {
                return hit;
            }

            let f = 1.0 / a;
            let s = ray.origin - v0;
            let u = f * dot(s, h);

            if (u < 0.0 || u > 1.0) {
                return hit;
            }

            let q = cross(s, e1);
            let v = f * dot(ray.direction, q);

            if (v < 0.0 || u + v > 1.0) {
                return hit;
            }

            let t = f * dot(e2, q);

            if (t >= ray.tmin && t <= ray.tmax) {
                hit.hit = true;
                hit.distance = t;
                hit.position = ray.origin + t * ray.direction;

                let w = 1.0 - u - v;
                hit.normal = normalize(w * n0 + u * n1 + v * n2);
                hit.materialIndex = materialIndex;
            }

            return hit;
        }

        fn intersectScene(ray: Ray) -> HitInfo {
            var hitInfo: HitInfo;
            hitInfo.hit = false;
            hitInfo.distance = 1000000.0;

            for (var i: u32 = 0; i < arrayLength(&indices); i++) {
                let index = indices[i];
                let v0 = vertices[index.x].xyz;
                let v1 = vertices[index.y].xyz;
                let v2 = vertices[index.z].xyz;
                let n0 = normals[index.x].xyz;
                let n1 = normals[index.y].xyz;
                let n2 = normals[index.z].xyz;
                let materialIndex = materialIndices[i];
                let triangleHit = intersectTriangle(ray, v0, v1, v2, n0, n1, n2, materialIndex);
                if (triangleHit.hit && triangleHit.distance < hitInfo.distance) {
                    hitInfo = triangleHit;
                }
            }

            return hitInfo;
        }

        fn isPointVisible(origin: vec3<f32>, ttarget: vec3<f32>) -> bool {
            var shadowRay: Ray;
            shadowRay.origin = origin + 0.002 * normalize(ttarget - origin);
            shadowRay.direction = normalize(ttarget - origin);
            shadowRay.tmin = 0.002;
            shadowRay.tmax = length(ttarget - origin) - 0.02;

            let shadowHit = intersectScene(shadowRay);
            return !shadowHit.hit;
        }

        fn sampleTriangle(v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>, r1: f32, r2: f32) -> vec3<f32> {
            let sqrtR1 = sqrt(r1);
            let alpha = 1.0 - sqrtR1;
            let beta = sqrtR1 * (1.0 - r2);
            let gamma = r2 * sqrtR1;
            return alpha * v0 + beta * v1 + gamma * v2;
        }

        fn sampleAreaLight(hitPosition: vec3<f32>, lightIndex: u32, seed: vec2<f32>) -> vec3<f32> {
            let lightTriIndex = lightIndices[lightIndex];
            let v0 = vertices[indices[lightTriIndex].x].xyz;
            let v1 = vertices[indices[lightTriIndex].y].xyz;
            let v2 = vertices[indices[lightTriIndex].z].xyz;


            let r1 = rand(seed);
            let r2 = rand(seed + vec2<f32>(1.0, 2.0));


            let lightPoint = sampleTriangle(v0, v1, v2, r1, r2);
            let lightNormal = normalize(cross(v1 - v0, v2 - v0));
            let lightArea = length(cross(v1 - v0, v2 - v0)) / 2.0;


            if (!isPointVisible(hitPosition, lightPoint)) {
                return vec3<f32>(0.0);
            }

            let lightDir = lightPoint - hitPosition;
            let distance = length(lightDir);
            let normalizedLightDir = normalize(lightDir);

            let cosTheta = max(dot(lightNormal, -normalizedLightDir), 0.0);

            let materialIndex = materialIndices[lightTriIndex];
            let emission = materials[materialIndex * 2 + 1].rgb;

            return emission * lightArea * cosTheta / (distance * distance);
        }

        @fragment
        fn fragmentMain(
            @builtin(position) fragPos: vec4<f32>,
            @location(0) uv: vec2<f32>
        ) -> @location(0) vec4<f32> {

            let seed = vec2<f32>(
                fragPos.x + uniforms.frameCount * 0.1,
                fragPos.y + uniforms.frameCount * 0.1
            );

            var pixelPos = vec2<f32>(
                fragPos.x + getRandom2D(seed).x,
                fragPos.y + getRandom2D(seed + vec2<f32>(1.0, 1.0)).y
            );

            let normalizedPos = vec2<f32>(
                (pixelPos.x / uniforms.width) * 2.0 - 1.0,
                (pixelPos.y / uniforms.height) * 2.0 - 1.0
            );

            let ray = generateRay(normalizedPos);
            let hitInfo = intersectScene(ray);

            var newColor: vec4<f32>;
            if (hitInfo.hit) {
                let materialIndex = hitInfo.materialIndex;
                let color = materials[materialIndex * 2].rgb;
                let emission = materials[materialIndex * 2 + 1].rgb;

                var directLight = vec3<f32>(0.0);
                for (var i: u32 = 0u; i < arrayLength(&lightIndices); i++) {

                    directLight += sampleAreaLight(hitInfo.position, i,
                        seed + vec2<f32>(f32(i) * 2.0, f32(i) * 2.0));
                }

                let finalColor = emission + color * directLight;
                newColor = vec4<f32>(finalColor, 1.0);
            } else {
                newColor = vec4<f32>(0.0, 0.0, 0.0, 1.0);
            }

            if (uniforms.frameCount > 0.0) {
                let prevColor = textureLoad(previousFrame, vec2<i32>(fragPos.xy), 0);
                return mix(prevColor, newColor, 1.0 / (uniforms.frameCount + 1.0));
            }
            return newColor;
        }
    `;

    const shaderModule = device.createShaderModule({
        code: shader
    });

    const pipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex: {
            module: shaderModule,
            entryPoint: 'vertexMain',
        },
        fragment: {
            module: shaderModule,
            entryPoint: 'fragmentMain',
            targets: [{
                format: canvasFormat
            }]
        },
        primitive: {
            topology: 'triangle-list',
        },
    });

    const textureSize = {
        width: canvas.width,
        height: canvas.height,
    };

    const renderTextures = [0, 1].map(() =>
        device.createTexture({
            size: textureSize,
            format: canvasFormat,
            usage: GPUTextureUsage.RENDER_ATTACHMENT |
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.COPY_SRC |
                GPUTextureUsage.COPY_DST,
        })
    );

    const uniformBuffer = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const toggleButton = document.getElementById('toggleButton');
    let currentTextureIndex = 0;
    let isProgressiveEnabled = true;
    let frameCount = 0;
    toggleButton.onclick = () => {
        isProgressiveEnabled = !isProgressiveEnabled;
        frameCount = 0;
        console.log(`Progressive Rendering is now ${isProgressiveEnabled ? 'enabled' : 'disabled'}`);
    };

    let aspectRatio = canvas.width / canvas.height;

    function updateUniforms() {
        const uniformData = new Float32Array([
            aspectRatio,
            frameCount,
            canvas.width,
            canvas.height
        ]);
        device.queue.writeBuffer(uniformBuffer, 0, uniformData);
    }

    function frame() {
        if (isProgressiveEnabled) {
            frameCount++;
        } else {
            frameCount = 0;
        }

        updateUniforms();

        const bindGroup = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: uniformBuffer,
                    },
                },
                {
                    binding: 1,
                    resource: {
                        buffer: vertexBuffer,
                    },
                },
                {
                    binding: 2,
                    resource: {
                        buffer: indexBuffer,
                    },
                },
                {
                    binding: 3,
                    resource: {
                        buffer: normalBuffer,
                    },
                },
                {
                    binding: 4,
                    resource: {
                        buffer: materialBuffer,
                    },
                },
                {
                    binding: 5,
                    resource: {
                        buffer: materialIndexBuffer,
                    },
                },
                {
                    binding: 6,
                    resource: {
                        buffer: lightIndicesBuffer,
                    },
                },
                {
                    binding: 7,
                    resource: renderTextures[1 - currentTextureIndex].createView(),
                },
            ],
        });

        const commandEncoder = device.createCommandEncoder();


        const renderPassEncoder = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: renderTextures[currentTextureIndex].createView(),
                clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
                loadOp: 'clear',
                storeOp: 'store',
            }]
        });

        renderPassEncoder.setPipeline(pipeline);
        renderPassEncoder.setBindGroup(0, bindGroup);
        renderPassEncoder.draw(6);
        renderPassEncoder.end();


        const canvasTexture = context.getCurrentTexture();
        const finalRenderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: canvasTexture.createView(),
                clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
                loadOp: 'clear',
                storeOp: 'store',
            }]
        });

        finalRenderPass.setPipeline(pipeline);
        finalRenderPass.setBindGroup(0, bindGroup);
        finalRenderPass.draw(6);
        finalRenderPass.end();

        if (isProgressiveEnabled) {
            currentTextureIndex = 1 - currentTextureIndex;
        }

        device.queue.submit([commandEncoder.finish()]);
        requestAnimationFrame(frame);
    }

    requestAnimationFrame(frame);
    cornellBoxData.materials.forEach((material, index) => {
        console.log(`Material ${index}:`, material);

    });
    console.log("Material Indices:", cornellBoxData.mat_indices.slice(0, 20));
    console.log("Material buffer size:", materialData.byteLength);
    console.log("Material index buffer size:", cornellBoxData.mat_indices.byteLength);
}

window.onload = init;
