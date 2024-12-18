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

            let fov = radians(60.0);
            let aspect = 1.0;
            let tanFov = tan(fov * 0.5);

            let x = uv.x * aspect * tanFov;
            let y = uv.y * tanFov;

            var ray: Ray;
            ray.origin = eye;
            ray.direction = normalize(-w + x*u + y*v);
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

        @fragment
        fn fragmentMain(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
            let ray = generateRay(uv);
            var hitInfo: HitInfo;
            hitInfo.hit = false;
            hitInfo.distance = 1000000.0;

            // Plane
            let planeHit = intersectPlane(ray, vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0));
            if (planeHit.hit && planeHit.distance < hitInfo.distance) {
                hitInfo = planeHit;
            }

            // Sphere
            let sphereHit = intersectSphere(ray, vec3<f32>(0.0, 0.5, 0.0), 0.3);
            if (sphereHit.hit && sphereHit.distance < hitInfo.distance) {
                hitInfo = sphereHit;
            }

            // Triangle
            let triangleHit = intersectTriangle(ray, vec3<f32>(-0.2, 0.1, 0.9), vec3<f32>(0.2, 0.1, 0.9), vec3<f32>(-0.2, 0.1, -0.1));
            if (triangleHit.hit && triangleHit.distance < hitInfo.distance) {
                hitInfo = triangleHit;
            }

            if (hitInfo.hit) {
                return vec4<f32>(hitInfo.color, 1.0);
            } else {
                return vec4<f32>(0.1, 0.3, 0.6, 1.0);  // Background color
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

    function frame() {
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
        renderPass.draw(6);
        renderPass.end();

        device.queue.submit([commandEncoder.finish()]);
        requestAnimationFrame(frame);
    }

    requestAnimationFrame(frame);
}

window.onload = init;