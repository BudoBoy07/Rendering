async function init() {
    if (!navigator.gpu) {
        alert("WebGPU not supported on this browser.");
        return;
    }

    const canvas = document.getElementById("myCanvas");
    canvas.width = 800;
    canvas.height = 450;
    const adapter = await navigator.gpu.requestAdapter();
    const device = await adapter.requestDevice();

    const context = canvas.getContext("webgpu");
    const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
        device: device,
        format: canvasFormat,
    });

    // Load the Utah teapot
    const teapotData = await readOBJFile('teapot.obj', 1.0, false);

    // Create and upload buffers
    const vertexBuffer = device.createBuffer({
        size: teapotData.vertices.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(vertexBuffer, 0, teapotData.vertices);

    const indexBuffer = device.createBuffer({
        size: teapotData.indices.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(indexBuffer, 0, teapotData.indices);

    // New: Create and upload normal buffer
    const normalBuffer = device.createBuffer({
        size: teapotData.normals.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(normalBuffer, 0, teapotData.normals);

    const shader = `
        struct Uniforms {
            aspectRatio: f32,
        }
        @group(0) @binding(0) var<uniform> uniforms: Uniforms;
        @group(0) @binding(1) var<storage, read> vertices: array<vec4<f32>>;
        @group(0) @binding(2) var<storage, read> indices: array<vec4<u32>>;
        @group(0) @binding(3) var<storage, read> normals: array<vec4<f32>>; // New: Normal buffer
        @group(0) @binding(4) var envMap: texture_2d<f32>;
        @group(0) @binding(5) var envSampler: sampler;
        @group(0) @binding(6) var<uniform> shadingMode: u32;  // Add this with the other uniforms


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
        }

        struct Light {
            direction: vec3<f32>,
            color: vec3<f32>,
        }

        struct Plane {
            normal: vec3<f32>,
            point: vec3<f32>,
        }

        fn intersectPlane(ray: Ray, plane: Plane) -> HitInfo {
            var hit: HitInfo;
            hit.hit = false;

            let denom = dot(ray.direction, plane.normal);
            if (abs(denom) > 0.0001) {
                let t = dot(plane.point - ray.origin, plane.normal) / denom;
                if (t >= ray.tmin && t <= ray.tmax) {
                    hit.hit = true;
                    hit.distance = t;
                    hit.position = ray.origin + t * ray.direction;
                    hit.normal = plane.normal;
                }
            }
            return hit;
        }

        fn computeAmbientOcclusion(position: vec3<f32>, normal: vec3<f32>, numSamples: u32) -> f32 {
            var occlusion = 0.0;
            let radius = 0.5;  // Reduced radius

            for (var i = 0u; i < numSamples; i++) {
                // Generate random direction in hemisphere
                let phi = (f32(i) / f32(numSamples)) * 2.0 * 3.14159;
                let cosTheta = sqrt(f32(i) / f32(numSamples));
                let sinTheta = sqrt(1.0 - cosTheta * cosTheta);

                let x = cos(phi) * sinTheta;
                let y = sin(phi) * sinTheta;
                let z = cosTheta;

                // Initialize the ray directly
                let sampleRay = Ray(
                    position + normal * 0.001,  // origin
                    normalize(vec3<f32>(x, y, z)),  // direction
                    0.001,  // tmin
                    radius  // tmax
                );

                let hit = intersectScene(sampleRay);
                if (hit.hit) {
                    occlusion += 1.0;
                }

                // Add early exit condition
                if (occlusion > f32(numSamples) * 0.8) {
                    return 0.2;  // Early exit if heavily occluded
                }
            }

            return 1.0 - (occlusion / f32(numSamples));
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
            let eye = vec3<f32>(0.15, 1.5, 10.0);
            let lookAt = vec3<f32>(0.15, 1.5, 0.0);
            let up = vec3<f32>(0.0, 1.0, 0.0);

            let w = normalize(eye - lookAt);
            let u = normalize(cross(up, w));
            let v = cross(w, u);

            let aspect = uniforms.aspectRatio;
            let d = 2.5;

            let x = uv.x * aspect;
            let y = uv.y;

            var ray: Ray;
            ray.origin = eye;
            ray.direction = normalize(x*u + y*v - d*w);
            ray.tmin = 0.001;
            ray.tmax = 1000.0;

            return ray;
        }

        fn reflectRay(ray: Ray, hitInfo: HitInfo) -> Ray {
            var reflectedRay: Ray;
            reflectedRay.origin = hitInfo.position + hitInfo.normal * 0.001;
            reflectedRay.direction = reflect(ray.direction, hitInfo.normal);
            reflectedRay.tmin = 0.001;
            reflectedRay.tmax = 1000.0;
            return reflectedRay;
        }

        fn intersectTriangle(ray: Ray, v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>, n0: vec3<f32>, n1: vec3<f32>, n2: vec3<f32>) -> HitInfo {
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

                // New: Interpolate normal using barycentric coordinates
                let w = 1.0 - u - v;
                hit.normal = normalize(w * n0 + u * n1 + v * n2);
            }

            return hit;
        }

        fn intersectScene(ray: Ray) -> HitInfo {
            var hitInfo: HitInfo;
            hitInfo.hit = false;
            hitInfo.distance = 1000000.0;

            // Add early exit conditions
            if (ray.tmax < ray.tmin) {
                return hitInfo;
            }

            // Add broad phase bounding box test here if possible

            // Optimize loop to process fewer triangles if possible
            let triangleCount = arrayLength(&indices);
            for (var i: u32 = 0u; i < triangleCount; i += 1u) {
                let index = indices[i];
                let v0 = vertices[index.x].xyz;
                let v1 = vertices[index.y].xyz;
                let v2 = vertices[index.z].xyz;
                let n0 = normals[index.x].xyz;
                let n1 = normals[index.y].xyz;
                let n2 = normals[index.z].xyz;
                let triangleHit = intersectTriangle(ray, v0, v1, v2, n0, n1, n2);
                if (triangleHit.hit && triangleHit.distance < hitInfo.distance) {
                    hitInfo = triangleHit;
                }
            }

            return hitInfo;
        }

        fn sampleDirectionalLight() -> Light {
            var light: Light;
            light.direction = normalize(vec3<f32>(-1.0));
            light.color = vec3<f32>(3.14159, 3.14159, 3.14159);
            return light;
        }

        fn directionToUV(dir: vec3<f32>) -> vec2<f32> {
            let u = 0.5 + atan2(dir.x, -dir.z) / (2.0 * 3.14159);
            let v = 0.5 - asin(dir.y) / 3.14159;
            return vec2<f32>(u, v);
        }

        fn sampleEnvironmentMap(direction: vec3<f32>) -> vec4<f32> {
            let uv = directionToUV(direction);
            let rgbe = textureSampleLevel(envMap, envSampler, uv, 0.0);

            // Decode HDR values
            var hdrColor = rgbe.rgb * pow(2.0, rgbe.a * 255.0 - 128.0);

            // Apply gamma correction (gamma = 2.2)
            hdrColor = pow(hdrColor, vec3<f32>(1.0/2.2));

            return vec4<f32>(hdrColor, 1.0);
        }

        @fragment
        fn fragmentMain(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
            let ray = generateRay(uv);
            let hitInfo = intersectScene(ray);
            let plane = Plane(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, -1.0, 0.0));
            let planeHit = intersectPlane(ray, plane);

            var color: vec4<f32>;

            if (hitInfo.hit && planeHit.hit) {
                // Both hit, choose the closer one
                if (hitInfo.distance < planeHit.distance) {
                    // Object hit is closer
                    switch(shadingMode) {
                        case 0u: {  // Base color
                            let light = sampleDirectionalLight();
                            let diffuse = vec3<f32>(0.9);
                            let lambert = max(dot(hitInfo.normal, -light.direction), 0.0);
                            color = vec4<f32>(diffuse * light.color * lambert, 1.0);
                        }
                        case 1u: {  // Mirror
                            let reflectedRay = reflectRay(ray, hitInfo);
                            color = sampleEnvironmentMap(reflectedRay.direction);
                        }
                        case 2u: {  // Diffuse
                            let light = sampleDirectionalLight();
                            let diffuse = vec3<f32>(0.7);
                            let lambert = max(dot(hitInfo.normal, -light.direction), 0.0);
                            let ambient = sampleEnvironmentMap(hitInfo.normal).xyz * 0.3;
                            color = vec4<f32>(diffuse * lambert + ambient, 1.0);
                        }
                        default: {
                            color = vec4<f32>(1.0, 0.0, 1.0, 1.0);
                        }
                    }
                } else {
                    // Plane hit is closer
                    let ao = computeAmbientOcclusion(planeHit.position, planeHit.normal, 8u);
                    color = vec4<f32>(vec3<f32>(ao), 1.0);
                }
            } else if (hitInfo.hit) {
                // Only object hit
                switch(shadingMode) {
                    case 0u: {  // Base color
                        let light = sampleDirectionalLight();
                        let diffuse = vec3<f32>(0.9);
                        let lambert = max(dot(hitInfo.normal, -light.direction), 0.0);
                        color = vec4<f32>(diffuse * light.color * lambert, 1.0);
                    }
                    case 1u: {  // Mirror
                        let reflectedRay = reflectRay(ray, hitInfo);
                        color = sampleEnvironmentMap(reflectedRay.direction);
                    }
                    case 2u: {  // Diffuse
                        let light = sampleDirectionalLight();
                        let diffuse = vec3<f32>(0.7);
                        let lambert = max(dot(hitInfo.normal, -light.direction), 0.0);
                        let ambient = sampleEnvironmentMap(hitInfo.normal).xyz * 0.3;
                        color = vec4<f32>(diffuse * lambert + ambient, 1.0);
                    }
                    default: {
                        color = vec4<f32>(1.0, 0.0, 1.0, 1.0);
                    }
                }
            } else if (planeHit.hit) {
                // Only plane hit
                let ao = computeAmbientOcclusion(planeHit.position, planeHit.normal, 16u);
                color = vec4<f32>(vec3<f32>(ao), 1.0);
            } else {
                // No hit, show environment
                color = sampleEnvironmentMap(ray.direction);
            }

            return color;
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
        size: 4,  // 1 float, 4 bytes
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const shadingModeBuffer = device.createBuffer({
        size: 4,  // 1 uint32, 4 bytes
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Load the HDR environment map
    const envMapImage = new HDRImage();
    envMapImage.src = 'luxo-pxr-campus.hdr.png';  // Use the HDR version
    await new Promise(resolve => envMapImage.onload = resolve);

    const envMapTexture = device.createTexture({
        size: [envMapImage.width, envMapImage.height, 1],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });

    device.queue.copyExternalImageToTexture(
        { source: envMapImage },
        { texture: envMapTexture },
        [envMapImage.width, envMapImage.height]
    );

    const envMapSampler = device.createSampler({
        magFilter: 'linear',
        minFilter: 'linear',
        mipmapFilter: 'linear',
        addressModeU: 'repeat',
        addressModeV: 'clamp-to-edge',
    });

    // Update your bind group creation
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
                resource: envMapTexture.createView(),
            },
            {
                binding: 5,
                resource: envMapSampler,
            },
            {
                binding: 6,
                resource: {
                    buffer: shadingModeBuffer,
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

    const sphereShaderSelect = document.getElementById('sphereShaderSelect');
    sphereShaderSelect.addEventListener('change', () => {
        const mode = parseInt(sphereShaderSelect.value);
        device.queue.writeBuffer(shadingModeBuffer, 0, new Uint32Array([mode]));
    });

    // Initialize the shading mode to match the default selected option
    device.queue.writeBuffer(shadingModeBuffer, 0, new Uint32Array([1])); // Start with mirror selected

    requestAnimationFrame(frame);
}

window.onload = init;