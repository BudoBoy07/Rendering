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

    const shader = `
        struct Uniforms {
            aspectRatio: f32,
            cameraConstant: f32,
        }
        @group(0) @binding(0) var<uniform> uniforms: Uniforms;

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
            color: vec3<f32>,
        }

        struct Light {
            position: vec3<f32>,
            intensity: vec3<f32>,
            distance: f32,
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
            let eye = vec3<f32>(2.0, 1.5, 2.0);
            let lookAt = vec3<f32>(0.0, 0.5, 0.0);
            let up = vec3<f32>(0.0, 1.0, 0.0);

            let w = normalize(eye - lookAt);
            let u = normalize(cross(up, w));
            let v = cross(w, u);

            let aspect = uniforms.aspectRatio;
            let d = uniforms.cameraConstant;

            let x = uv.x * aspect;
            let y = uv.y;

            var ray: Ray;
            ray.origin = eye;
            ray.direction = normalize(x*u + y*v - d*w);
            ray.tmin = 0.001;
            ray.tmax = 1000.0;

            return ray;
        }

        fn intersectPlane(ray: Ray, position: vec3<f32>, normal: vec3<f32>) -> HitInfo {
            var hit: HitInfo;
            hit.hit = false;

            let denom = dot(ray.direction, normal);
            if (abs(denom) > 0.0001) {
                let t = dot(position - ray.origin, normal) / denom;
                if (t >= ray.tmin && t <= ray.tmax) {
                    hit.hit = true;
                    hit.distance = t;
                    hit.position = ray.origin + t * ray.direction;
                    hit.normal = normal;
                    hit.color = vec3<f32>(0.1, 0.7, 0.0);
                }
            }

            return hit;
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
                    hit.color = vec3<f32>(0.0, 0.0, 0.0);
                }
            }

            return hit;
        }

        fn intersectTriangle(ray: Ray, v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>) -> HitInfo {
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
                hit.normal = normalize(cross(e1, e2));
                hit.color = vec3<f32>(0.4, 0.3, 0.2);
            }

            return hit;
        }

        fn intersectScene(ray: Ray) -> HitInfo {
            var hitInfo: HitInfo;
            hitInfo.hit = false;
            hitInfo.distance = 1000000.0;


            let planeHit = intersectPlane(ray, vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0));
            if (planeHit.hit && planeHit.distance < hitInfo.distance) {
                hitInfo = planeHit;
            }


            let sphereHit = intersectSphere(ray, vec3<f32>(0.0, 0.5, 0.0), 0.3);
            if (sphereHit.hit && sphereHit.distance < hitInfo.distance) {
                hitInfo = sphereHit;
            }


            let triangleHit = intersectTriangle(ray, vec3<f32>(-0.2, 0.1, 0.9), vec3<f32>(0.2, 0.1, 0.9), vec3<f32>(-0.2, 0.1, -0.1));
            if (triangleHit.hit && triangleHit.distance < hitInfo.distance) {
                hitInfo = triangleHit;
            }

            return hitInfo;
        }

        fn samplePointLight(position: vec3<f32>) -> Light {
            var light: Light;
            light.position = vec3<f32>(0.0, 1.0, 0.0);
            light.intensity = vec3<f32>(3.14159, 3.14159, 3.14159);
            light.distance = length(light.position - position);
            return light;
        }

        fn shade(hit: HitInfo, ray: Ray) -> vec3<f32> {
            let light = samplePointLight(hit.position);
            let lightDir = normalize(light.position - hit.position);


            var shadowRay: Ray;
            shadowRay.origin = hit.position + hit.normal * 0.001;
            shadowRay.direction = lightDir;
            shadowRay.tmin = 0.001;
            shadowRay.tmax = light.distance;

            let shadowHit = intersectScene(shadowRay);

            let ambient = 0.1 * hit.color;

            if (shadowHit.hit) {
                return ambient;
            }

            let diffuse = max(dot(hit.normal, lightDir), 0.0);
            let r2 = dot(light.position - hit.position, light.position - hit.position);
            let attenuation = 1.0 / r2;

            let diffuseColor = 0.9 * hit.color * diffuse * attenuation * light.intensity;

            return ambient + diffuseColor;
        }

        @fragment
        fn fragmentMain(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
            let ray = generateRay(uv);
            let hitInfo = intersectScene(ray);

            if (hitInfo.hit) {
                let shadedColor = shade(hitInfo, ray);
                return vec4<f32>(shadedColor, 1.0);
            } else {
                return vec4<f32>(0.1, 0.3, 0.6, 1.0);
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
        size: 8,
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
        ],
    });

    let aspectRatio = canvas.width / canvas.height;
    let cameraConstant = 1.0;

    function updateUniforms() {
        const uniformData = new Float32Array([aspectRatio, cameraConstant]);
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


    const zoomSlider = document.getElementById('zoomSlider');
    zoomSlider.addEventListener('input', function() {
        cameraConstant = parseFloat(this.value);
    });


    window.addEventListener('resize', function() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        aspectRatio = canvas.width / canvas.height;
        context.configure({
            device: device,
            format: canvasFormat,
            size: [canvas.width, canvas.height]
        });
    });

    requestAnimationFrame(frame);
}

window.onload = init;