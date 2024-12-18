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

    const teapotData = await readOBJFile('teapot.obj', 1.0, false);


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
        @group(0) @binding(1) var envMap: texture_2d<f32>;
        @group(0) @binding(2) var envSampler: sampler;

        struct VertexOutput {
            @builtin(position) position: vec4<f32>,
            @location(0) uv: vec2<f32>,
        }

        struct Material {
            albedo: vec3<f32>,
            metallic: f32,
            roughness: f32,
            specular: f32,
        }

        fn calculateMaterial(hitInfo: HitInfo, viewDir: vec3<f32>) -> vec4<f32> {
            let light = sampleDirectionalLight();

            var material: Material;
            let sphereIndex = floor((hitInfo.position.x + 5.0) / 2.0);

            switch(i32(sphereIndex)) {
                case 0: { //Chrome-like
                    material.albedo = vec3<f32>(0.95, 0.95, 0.95);
                    material.metallic = 1.0;
                    material.roughness = 0.1;
                    material.specular = 0.8;
                }
                case 1: { //Gold-like
                    material.albedo = vec3<f32>(1.0, 0.8, 0.3);
                    material.metallic = 0.6;
                    material.roughness = 0.14;
                    material.specular = 0.5;
                }
                case 2: { //Ceramic-like
                    material.albedo = vec3<f32>(0.9, 0.9, 0.9);
                    material.metallic = 0.1;
                    material.roughness = 0.01;
                    material.specular = 0.5;
                }
                case 3: { //Emerald-like
                    material.albedo = vec3<f32>(0.2, 0.8, 0.2);
                    material.metallic = 1.9; //this should not be over 1, but it looks cool
                    material.roughness = 0.1;
                    material.specular = 0.3;
                }
                default: { //Rubber-like
                    material.albedo = vec3<f32>(0.1, 0.1, 0.1);
                    material.metallic = 0.0;
                    material.roughness = 0.8;
                    material.specular = 0.1;
                }
            }

            let N = hitInfo.normal;
            let V = -viewDir;
            let L = -light.direction;
            let H = normalize(L + V);

            let NdotL = max(dot(N, L), 0.0);
            let NdotH = max(dot(N, H), 0.0);
            let NdotV = max(dot(N, V), 0.0);

            let R = reflect(viewDir, N);

            let envColor = sampleEnvironmentMap(R).xyz;

            //diffuse lighting
            let diffuse = material.albedo * NdotL;

            //specular with roughness
            let specPower = 2.0 / pow(material.roughness + 0.001, 4.0) - 2.0;
            let specular = material.specular * pow(NdotH, specPower);

            //fresnel effect
            let F0 = mix(vec3<f32>(0.04), material.albedo, material.metallic);
            let fresnel = F0 + (1.0 - F0) * pow(1.0 - NdotV, 5.0);

            //mix direct lighting with environment reflections
            let directLighting = (diffuse + specular) * light.color;
            let envReflection = envColor * fresnel;

            //final mix based on metallic and roughness
            let reflectivity = (1.0 - material.roughness) * (0.5 + 0.5 * material.metallic);
            let finalColor = mix(directLighting, envReflection, reflectivity);

            return vec4<f32>(finalColor, 1.0);
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


                let w = 1.0 - u - v;
                hit.normal = normalize(w * n0 + u * n1 + v * n2);
            }

            return hit;
        }

        fn intersectScene(ray: Ray) -> HitInfo {
            var hitInfo: HitInfo;
            hitInfo.hit = false;
            hitInfo.distance = 1000000.0;

            let sphere0 = intersectSphere(ray, vec3<f32>(-4.0, -1.1, 0.4), 1.0);
            let sphere1 = intersectSphere(ray, vec3<f32>(-2.0, -1.1, 0.2), 1.0);
            let sphere2 = intersectSphere(ray, vec3<f32>(0.0, -1.1, 0.0), 1.0);
            let sphere3 = intersectSphere(ray, vec3<f32>(2.0, -1.1, -0.2), 1.0);
            let sphere4 = intersectSphere(ray, vec3<f32>(4.0, -1.1, -0.4), 1.0);

            if (sphere0.hit && sphere0.distance < hitInfo.distance) {
                hitInfo = sphere0;
            }
            if (sphere1.hit && sphere1.distance < hitInfo.distance) {
                hitInfo = sphere1;
            }
            if (sphere2.hit && sphere2.distance < hitInfo.distance) {
                hitInfo = sphere2;
            }
            if (sphere3.hit && sphere3.distance < hitInfo.distance) {
                hitInfo = sphere3;
            }
            if (sphere4.hit && sphere4.distance < hitInfo.distance) {
                hitInfo = sphere4;
            }

            /*teapot
            for (var i: u32 = 0; i < arrayLength(&indices); i++) {
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
            */

            return hitInfo;
        }

        fn intersectSphere(ray: Ray, center: vec3<f32>, radius: f32) -> HitInfo {
            var hit: HitInfo;
            hit.hit = false;

            let oc = ray.origin - center;
            let a = dot(ray.direction, ray.direction);
            let b = 2.0 * dot(oc, ray.direction);
            let c = dot(oc, oc) - radius * radius;
            let discriminant = b * b - 4.0 * a * c;

            if (discriminant > 0.0) {
                let t = (-b - sqrt(discriminant)) / (2.0 * a);
                if (t >= ray.tmin && t <= ray.tmax) {
                    hit.hit = true;
                    hit.distance = t;
                    hit.position = ray.origin + t * ray.direction;
                    hit.normal = normalize(hit.position - center);
                    return hit;
                }
            }

            return hit;
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
            return textureSampleLevel(envMap, envSampler, uv, 0.0);
        }

        @fragment
        fn fragmentMain(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
            let ray = generateRay(uv);
            let hitInfo = intersectScene(ray);

            if (hitInfo.hit) {
                return calculateMaterial(hitInfo, ray.direction);
            } else {
                return sampleEnvironmentMap(ray.direction);
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

    const shadingModeBuffer = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });


    const envMapImage = new Image();
    envMapImage.src = 'Floor_Panorama.jpg';
    await envMapImage.decode();

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
                resource: envMapTexture.createView(),
            },
            {
                binding: 2,
                resource: envMapSampler,
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