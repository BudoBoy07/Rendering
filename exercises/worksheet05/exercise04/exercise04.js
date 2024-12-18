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


    const materialIndexBuffer = device.createBuffer({
        size: cornellBoxData.mat_indices.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(materialIndexBuffer, 0, cornellBoxData.mat_indices);

    const shader = `
        struct Uniforms {
            aspectRatio: f32,
        }
        @group(0) @binding(0) var<uniform> uniforms: Uniforms;
        @group(0) @binding(1) var<storage, read> vertices: array<vec4<f32>>;
        @group(0) @binding(2) var<storage, read> indices: array<vec4<u32>>;
        @group(0) @binding(3) var<storage, read> normals: array<vec4<f32>>;
        @group(0) @binding(4) var<storage, read> materials: array<vec4<f32>>;
        @group(0) @binding(5) var<storage, read> materialIndices: array<u32>;

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
            let up = vec3<f32>(0.0, 1.0, 0.0);

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

    @fragment
    fn fragmentMain(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
        let ray = generateRay(uv);
        let hitInfo = intersectScene(ray);

        if (hitInfo.hit) {
            let materialIndex = hitInfo.materialIndex;
            let color = materials[materialIndex * 2].rgb;
            let emission = materials[materialIndex * 2 + 1].rgb;
            return vec4<f32>(color + emission, 1.0);

        } else {
            return vec4<f32>(0.0, 0.0, 0.0, 1.0);

        }
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

    const uniformBuffer = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

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
        ],
    });

    let aspectRatio = canvas.width / canvas.height;

    function updateUniforms() {
        const uniformData = new Float32Array([aspectRatio]);
        device.queue.writeBuffer(uniformBuffer, 0, uniformData);
    }

    function frame() {
        updateUniforms();

        const commandEncoder = device.createCommandEncoder();
        const textureView = context.getCurrentTexture().createView();

        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: textureView,
                clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
                loadOp: 'clear',
                storeOp: 'store',
            }]
        });

        renderPass.setPipeline(pipeline);
        renderPass.setBindGroup(0, bindGroup);
        renderPass.draw(6);
        renderPass.end();

        device.queue.submit([commandEncoder.finish()]);
        requestAnimationFrame(frame);
    }

    requestAnimationFrame(frame);
}

window.onload = init;
