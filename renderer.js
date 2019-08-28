var Renderer = BaseObject("Renderer", false);
{
	var rasterVertShader = `#version 300 es
		layout(location=0) in vec3 a_position;

		out vec2 v_texcoord;

		void main(){
			v_texcoord = (a_position.xy + 1.0) / 2.0;
			gl_Position = vec4(a_position, 1.0);
		}
	`;

	var rasterFragShader = `#version 300 es
		precision mediump float;

		in vec2 v_texcoord;

		uniform sampler2D uTexture;

		out vec4 fragColour;

		void main(){
			fragColour = texture(uTexture, v_texcoord);
		}
	`;

	var filterVertShader = `#version 300 es
		layout(location=0) in vec3 a_position;

		out vec2 v_texcoord;

		void main(){
			v_texcoord = (a_position.xy + 1.0) / 2.0;
			gl_Position = vec4(a_position, 1.0);
		}
	`;

	var filterFragShader = `#version 300 es
		precision mediump float;

		in vec2 v_texcoord;

		uniform sampler2D uLastScreen;
		uniform sampler2D uCurrentScreen;

		out vec4 fragColour;

		struct PixelGradient {
			vec3 hsv;
			vec2 dH;
			vec2 dS;
			vec2 dV;
		};

		/* This code was found here: http://lolengine.net/blog/2013/07/27/rgb-to-hsv-in-glsl */
		vec3 rgbToHsv(vec3 c){
			vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
		    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
		    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

		    float d = q.x - min(q.w, q.y);
		    float e = 1.0e-10;
		    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
		}

		vec3 hsvToRgb(vec3 c)
		{
		    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
		    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
		    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
		}

		PixelGradient calculateGradient(vec4 pixelVal){
			vec3 hsv = rgbToHsv(pixelVal.rgb);

			vec3 dx = dFdx(hsv);
			vec3 dy = dFdy(hsv);

			return PixelGradient(
				hsv,
				vec2(dx.r, dy.r), 
				vec2(dx.g, dy.g), 
				vec2(dx.b, dy.b));
		}

		float mixComponents(float a, float b, vec2 dA, vec2 dB){
			float mixVal = abs(dot(dA, dB)) * 0.5;

			return isnan(mixVal) ? a : mix(a, b, mixVal);
		}

		void main(){
			vec4 last = texture(uLastScreen, v_texcoord);
			vec4 curr = texture(uCurrentScreen, v_texcoord);

			PixelGradient gLast = calculateGradient(last);
			PixelGradient gCurr = calculateGradient(curr);

			vec3 hsv = vec3(
				gLast.hsv.r,
				gLast.hsv.g,
				mix(gLast.hsv.b, gCurr.hsv.b, 0.5)
			);

			fragColour = vec4(
				hsvToRgb(hsv),
				1.0);
		}
	`;

	var vertexShader = `#version 300 es
	layout(location=0) in vec3 a_position;

	uniform mat4 uMVMatrix;
	//uniform vec3 uViewingAngle;
	//uniform float uAspectRatio;

	out vec2 v_texcoord;
	out vec3 v_ray;
	out vec4 v_eyePos;

	float viewAngle = radians(180.9 / 2.0);
	float aspectRatio = 1.0;

	void main(){
		v_texcoord = (a_position.xy + 1.0) / 2.0;
		v_eyePos = uMVMatrix * vec4(0.0, 0.0, 0.0, 1.0);
		v_ray = (uMVMatrix * vec4(a_position.xy, 1.0, 1.0)).xyz - v_eyePos.xyz;
		gl_Position = vec4(a_position, 1.0);
	}
	`;

	var inputProperties = `
		in vec2 v_texcoord;
		in vec3 v_ray;
		in vec4 v_eyePos;

		uniform Shape s1;
		uniform Shape s2;
		uniform Shape s3;
		
		uniform float uSeed;
	`

	/*var totalRands = 1223;
	var randValues = new Float32Array(totalRands);

	for(var i = 0; i < totalRands; ++i){
		randValues[i] = Math.random();
	}

	const float RAND_VALS[` + totalRands + `] = float[` + totalRands + `](` + randValues.join(',') + `);
	 Static's make no sense in the shader language and can't be used like this.
	*/

	var globalFunctions = `
	#define PI 3.1415926535897932384626433832795

	const vec2 RAND_BASE = vec2(12.9898,78.233);

	float rand(vec2 co){
    	return fract(sin(dot(co.xy, RAND_BASE)) * 43758.5453);
	}

	vec3 rand3(vec3 co){
		vec3 result = vec3(dot(co.yz, RAND_BASE), dot(co.xz, RAND_BASE), dot(co.xy, RAND_BASE));
		return fract(sin(result) * 43758.5453);
	}

	float boxSDF(vec3 p, Shape s){
		return length(max(abs(p - s.location) - s.r, 0.0));
	}

	float sphereSDF(vec3 p, Shape s){
		return length(p - s.location) - s.r;
	}

	float planeSDF(vec3 p, Shape s){
		return dot(p, s.location) + s.r;
	}

	float shapeSDF(vec3 point, Shape shape){
		switch(shape.type){
			case 1:
				return planeSDF(point, shape);

			case 2:
				return boxSDF(point, shape);

			default:
				return sphereSDF(point, shape);
		}

		return MAXDIST;
	}

	Shape getShapeAtPoint(vec3 point){
		float r = shapeSDF(point, s1);

		if(r < EPSILON){
			return s1;
		}

		r = shapeSDF(point, s2);

		if(r < EPSILON){
			return s2;
		}

		return s3;
	}

	float sceneSDF(vec3 point){
		float r = shapeSDF(point, s1);
		r = min(r, shapeSDF(point, s2));
		r = min(r, shapeSDF(point, s3));

		return r;
	}

	// TODO: Calculate the normal for cube
	vec3 estimateNormal(vec3 p, Shape s){
		vec3 n;
		switch(s.type){
			case 0:
				n = normalize(p - s.location);
				break;

			case 1:
				n = normalize(s.location);
				break;

			default:
				n = normalize(vec3(
					sceneSDF(vec3(p.x + EPSILON, p.y, p.z)) - sceneSDF(vec3(p.x - EPSILON, p.y, p.z)),
					sceneSDF(vec3(p.x, p.y + EPSILON, p.z)) - sceneSDF(vec3(p.x, p.y - EPSILON, p.z)),
					sceneSDF(vec3(p.x, p.y, p.z + EPSILON)) - sceneSDF(vec3(p.x, p.y, p.z - EPSILON))
				));
		}

		return n;
	}

	vec4 castRay(vec3 eye, vec3 ray, float dist, float maxDist){
		float d = dist;
		vec4 p = vec4(0.0, 0.0, 0.0, 0.0);
		float sdf = 0.0;

		for(int i = 0; i < 200 && d < maxDist; ++i){
			p.xyz = (ray * d) + eye;
			sdf = sceneSDF(p.xyz);

			if(sdf < EPSILON){
				p.w = d;
				return p;
			}else{
				d += sdf;
			}
		}

		p.w = d;
		return p;
	}

	vec3 calculateReflection(vec3 eyePos, vec3 p, vec3 n, float roughness){
		// Calculate a reflection based on the surfaces glossiness
		vec3 e = normalize(p - eyePos);
		vec3 r = normalize(n + (((rand3(n + e + uSeed) * 2.0) - 1.0) * roughness));
		return reflect(e, r);
	}

	vec2 caculateCubeUV(vec3 p, Shape s){
		vec3 n = (p - s.location) / s.r;
		vec3 aN = abs(n);
		vec2 r = p.xy;

		if(aN.x > aN.y){
			if(aN.x > aN.z){
				r = p.yz;
			}
		}else if(aN.y > aN.z){
			r = p.xz;
		}

		return (r + 1.0) * 0.5;
	}

	vec2 calculateUV(vec3 p, Shape s){
		vec3 n;

		switch(s.type){
			case 0:
				n = normalize(p - s.location);
				return vec2(atan(n.x, n.z) / (2.0 * PI) + 0.5, n.y * 0.5 + 0.5);

			case 1:
				return mod(p.xz * 0.03, 1.0); //TODO: Use a material scale insead

			case 2:
				return caculateCubeUV(p, s);

			default:
				return p.xz;
		}
	}
	`

	var definedProperties = `
		#define BLUR 0.005
		#define EPSILON 0.00001
		#define MAXDIST 500.0
		#define MINDIST 0.00005
	`

	var objectTypes = `
	struct Shape {
		float r;
		vec3 location;
		int type; //0 - sphere, 1 - plane, 2 - box
		int materialId;
	};

	struct Material {
		int id;
		vec3 diffuseColour;
		vec3 specularColour;
		float roughness;
		float reflectionAmount;
		sampler2D diffuseTexture;
		sampler2D normalTexture;
		sampler2D roughnessTexture;
	};

	struct Light {
		vec3 colour;
		vec3 location;
		float power;
		float softFactor; //Higher the number, the sharper the shadow
		bool castShadow;
		int type; //0 - directional, 1 - point
	};
	`;

	/*const Shape s1 = Shape(3.0, vec3(0.0, 0.0, 0.0), 2, 1);
	const Shape s2 = Shape(20.0, vec3(20.0, 20.0, 20.0), 0, 2);
	const Shape s3 = Shape(5.0, vec3(0.0, 1.0, 0.0), 1, 3);
	`;*/ // Scene definition is temporary.

	var calculateMaterialShader = `#version 300 es
		precision mediump float;

		` + objectTypes
		+ inputProperties + `

		uniform vec3 sunAngle;

		uniform sampler2D rayDepth;
		uniform sampler2D eyeOffset;
		uniform sampler2D sceneRays;
		uniform sampler2D flatNormals;
		uniform sampler2D materialId;
		uniform sampler2D uvMap;
		uniform Material material;

		layout(location=0) out vec4 diffuseColour;
		layout(location=1) out vec4 ambientColour;
		layout(location=2) out vec4 normals;
		layout(location=3) out vec4 specularColour;
		` + definedProperties
		+ globalFunctions + `

		vec3 calculateAmbientLight(vec3 diffuseColour){
			return diffuseColour * 0.05;
		}

		vec3 getSkyboxColourCheck(vec3 ray, vec3 eyePos, vec3 lightAngle){
			float r, g, b;
			float cosRayUp = dot(vec3(0.0, 1.0, 0.0), ray);
			float cosLightUp = dot(vec3(0.0, 1.0, 0.0), -lightAngle);
			vec3 A = vec3(1.0, 0.0, 0.0);
			vec3 B = vec3(0.0, 0.5, 1.0);
			vec3 C = vec3(0.0, 0.0, 1.0);
			vec3 D = vec3(0.0, 0.0, 0.2);

			// Rayleigh Scattering
			vec3 skyColour;

			if(cosRayUp < cosLightUp){
				skyColour = mix(A, B, cosRayUp);
			}else{
				skyColour = mix(C, D, cosRayUp);
			}

			// Mie Scattering
			r = dot(lightAngle, -ray);
			g = pow(r, 1.5);
			b = pow(b, 2.5);

			vec3 sunColour = vec3(r, g, b);

			return skyColour;// + sunColour;
		}

		vec3 getSkyboxColourWithSun(vec3 ray, vec3 eyePos, vec3 lightAngle){

			vec3 redRay = normalize(vec3(ray.x, lightAngle.y * 1.0, ray.z));
			vec3 greenRay = normalize(vec3(ray.x, lightAngle.y * 0.0, ray.z));
			vec3 blueRay = normalize(vec3(ray.x, lightAngle.y * -1.0, ray.z));

			vec3 skyColour = vec3(dot(redRay, ray), dot(greenRay, ray), dot(blueRay, ray));

			float d = dot(ray, -lightAngle);
			float r = max((d - 0.7) / 0.7, 0.0);
			float g = max((d - 0.8) / 0.8, 0.0);
			float b = max((d - 0.95) / 0.95, 0.0);

			vec3 sunColour = vec3(r, g, b);

			return skyColour + sunColour;
		}

		vec3 getSkyboxColour(vec3 ray, vec3 eyePos, vec3 lightAngle){
			vec3 atmosphere = vec3(0.6, 0.6, 1.0);
			vec3 sky = vec3(0.0, 0.0, 0.5);
			vec3 ground = vec3(0.3, 0.3, 0.3);

			if(ray.y < 0.0){
				return mix(atmosphere, ground, -ray.y);
			}

			return mix(atmosphere, sky, ray.y);
		}

		vec3 calculateNormal(vec3 p, vec3 ray, vec2 uv){
			vec3 n = normalize((texture(flatNormals, v_texcoord)).rgb); // Normal
			vec3 t = cross(n, ray); // Tangent
			vec3 b = cross(t, n); // Bit Tangent
			vec3 offset = normalize((texture(material.normalTexture, uv).rgb * 2.0) - 1.0);

			mat3 transformMat = mat3(t, b, n);


			return vec3(transformMat * offset);
		}

		void main(){
			vec4 ray = texture(sceneRays, v_texcoord);

			if(ray.w <= 0.0){
				discard;
			}

			ray.xyz = normalize(ray.xyz);
			float d = texture(rayDepth, v_texcoord).r;

			if(d >= MAXDIST){
				vec3 light = normalize(sunAngle);
				ambientColour.rgb = getSkyboxColour(ray.xyz, v_eyePos.xyz, light) * ray.w;
				ambientColour.a = 1.0;
				return;
				//discard;
			}

			if(int(round(texture(materialId, v_texcoord).r * 255.0)) != material.id){
				discard;
			}

			vec3 eye = v_eyePos.xyz + texture(eyeOffset, v_texcoord).xyz;
			vec3 rayPoint = (ray.xyz * d) + eye;

			vec2 uv = texture(uvMap, v_texcoord).xy;

			normals = vec4(calculateNormal(rayPoint.xyz, ray.xyz, uv), material.reflectionAmount);
			vec3 dColour = texture(material.diffuseTexture, uv).rgb * material.diffuseColour;

			ambientColour = vec4(calculateAmbientLight(dColour.rgb) * ray.w, 1.0);
			diffuseColour = vec4(dColour, 1.0);
			specularColour = vec4(material.specularColour, material.roughness * texture(material.roughnessTexture, uv).r);
		}
	`;

	var calcAmbientBounceShader = `#version 300 es
		precision mediump float;

		` + objectTypes
		+ inputProperties + `
		uniform sampler2D eyeOffset;
		uniform sampler2D sceneRays;

		layout(location=0) out float rayDepth;
		` + definedProperties
		+ globalFunctions + `
		void main(){
			//if(rand(v_texcoord + uSeed) < 0.8){
			//	discard;
			//}

			vec4 ray = texture(sceneRays, v_texcoord);

			if(ray.w <= 0.0){
				discard;
			}

			vec4 rayPoint = castRay(v_eyePos.xyz + texture(eyeOffset, v_texcoord).xyz, normalize(ray.xyz), MINDIST, MAXDIST);

			if(rayPoint.w >= MAXDIST){
				discard;
			}

			rayDepth = rayPoint.w;
		}
	`

	//TODO: We'll need to add a ray start texture. This will allow us to perform the same calculations on reflections and refractions.
	var calcFrameShader = `#version 300 es
		precision mediump float;

		` + objectTypes 
		+ inputProperties + `

		uniform sampler2D rayDepth;
		uniform sampler2D normals;
		uniform Light light;
		uniform sampler2D eyeOffset;
		uniform sampler2D sceneRays;

		uniform sampler2D diffuseColour;
		uniform sampler2D specularColour;

		layout(location=0) out vec4 fragColour;

		` + definedProperties
		+ globalFunctions + `
		float getShadow(vec3 p, vec3 l, float minD, float maxD, float k){
			float result = 1.0;
			float d = minD;

			for(int i = 0; i < 20 && d < maxD; ++i){
				float h = sceneSDF((l * d) + p);

				if(h < EPSILON){
					return 0.0;
				}

				result = min(result, k * h/d);
				d += h;
			}

			return result;
		}

		float calculateSpecularAmount(vec3 e, vec3 l, vec3 n, float roughness){
			vec3 rl = normalize(reflect(l, n));
			float sPower = acos(dot(rl, -e));
			float val = roughness > 0.0 ? sPower / roughness : sPower; // Should be INF

			return exp(-(val * val));
		}

		vec3 getDirectionalLightColour(vec3 l, vec3 colour, vec3 p, vec3 n, vec3 e, float k){
			vec3 diff, spec;
			float d = getShadow(p, -l, EPSILON + 0.001, MAXDIST, k);

			if(d > 0.0){
				float nl = dot(n,-l);

				if(nl > 0.0){
					diff = nl * texture(diffuseColour, v_texcoord).rgb;

					// Use Gaussian Specular
					// TODO: Calculate specular for cases where we are inline with the light.
					vec4 specColour = vec4(0.0, 0.0, 0.0, 0.0);//texture(specularColour, v_texcoord);
					spec = calculateSpecularAmount(e, l, n, specColour.w) * specColour.rgb;
					
					return (diff + spec) * colour * d;
				}
			}

			return vec3(0.0, 0.0, 0.0);
		}

		vec3 getPointLightColour(vec3 l, vec3 colour, vec3 p, vec3 n, vec3 e, vec3 k){
			return vec3(0.0, 0.0, 0.0);
		}

		vec3 calculateLight(Light l, vec3 p, vec3 n, vec3 e){
			switch(l.type){
				case 0:
					return getDirectionalLightColour(normalize(l.location), l.colour, p, n, e, l.softFactor);

				default:
					return vec3(0.0, 0.0, 0.0);
			}
		}

		void main(){
			vec4 ray = texture(sceneRays, v_texcoord);

			if(ray.w <= 0.0){
				discard;
			}

			vec3 n = texture(normals, v_texcoord).xyz;

			if(length(n) <= 0.0){
				discard;
			}

			float d = texture(rayDepth, v_texcoord).r;
			vec3 eye = v_eyePos.xyz + texture(eyeOffset, v_texcoord).xyz;
			vec3 rayPoint = (normalize(ray.xyz) * d) + eye;

			n = normalize(n);

			fragColour = vec4(calculateLight(light, rayPoint, n, normalize(rayPoint - eye)), 1.0) * ray.w;
			//fragColour = texture(normals, v_texcoord);
		}
	`

	var calcReflectionsShader = `#version 300 es
		precision mediump float;

		` + objectTypes 
		+ inputProperties+ `

		uniform sampler2D currentOffset;
		uniform sampler2D currentRay;
		uniform sampler2D normals;
		uniform sampler2D specularColour;
		uniform int bounce;
		
		layout(location=0) out vec3 pointOffset;
		layout(location=1) out vec4 rayDirection;
		layout(location=2) out float rayDepth;
		layout(location=3) out float materialId; 
		layout(location=4) out vec4 flatNormalTexture;
		layout(location=5) out vec3 uvMap;
		` + definedProperties
		+ globalFunctions + `


		void main(){
			vec4 rayPoint;
			rayDepth = MAXDIST;

			if(bounce == 0){
				pointOffset = vec3(0.0);
				rayDirection = vec4(normalize(v_ray.xyz), 1.0);
			}else{
				vec4 ray = texture(currentRay, v_texcoord);

				if(ray.w == 0.0){
					return;
				}

				vec3 eye = v_eyePos.xyz + texture(currentOffset, v_texcoord).xyz;
				rayPoint = castRay(eye, normalize(ray.xyz), 0.0, MAXDIST);

				if(rayPoint.w >= MAXDIST){
					return;
				}

				vec4 normalVals = texture(normals, v_texcoord);
				vec3 n = normalize(normalVals.xyz);

				rayDirection = vec4(0.0, 0.0, 0.0, 0.0);
				float power = ray.w * normalVals.w;
				if(power > 0.0){
					float roughness = texture(specularColour, v_texcoord).w;
					rayDirection = vec4(normalize(calculateReflection(eye, rayPoint.xyz, n, roughness)), power);
				}

				pointOffset = rayPoint.xyz - v_eyePos.xyz;
			}

			if(rayDirection.w <= 0.0){
				return;
			}else{
				rayPoint = castRay(v_eyePos.xyz + pointOffset.xyz, normalize(rayDirection.xyz), MINDIST, MAXDIST);
				rayDepth = rayPoint.w;

				Shape s = getShapeAtPoint(rayPoint.xyz);
				materialId = float(s.materialId) / 255.0;

				uvMap = vec3(calculateUV(rayPoint.xyz, s), 0.0);

				flatNormalTexture.xyz = estimateNormal(rayPoint.xyz, s);
				flatNormalTexture.w = 1.0;
			}
		}
	`

	/* BOUNCE Method
	for(int b = 2; b >= 0 && amount > EPSILON; --b){
			rayPoint = castRay(p, ray, startPoint, remainingDist);

			if(rayPoint.w < 0.5){
				break;
			}

			s = getShapeAtPoint(rayPoint.xyz);
			result = calculatePixel(v_eyePos.xyz, rayPoint.xyz, s);
			fragColour.rgb += result.colour * amount;

			if(s.refAmount <= 0.0){
				break;
			}

			p = rayPoint.xyz;
			ray = calculateReflection(v_eyePos.xyz, p, s, result.normal);
			amount = (s.refAmount * amount);
			startPoint = EPSILON + 0.001;
		}
	*/

	function initWebGL(gl){
		gl.clearColor(0.0, 0.0, 0.0, 1.0);
		//gl.enable(gl.DEPTH_TEST);
		//gl.enable(gl.BLEND);
		//gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
		gl.blendFunc(gl.ONE, gl.ONE);
		
		var ext = gl.getExtension("EXT_color_buffer_float");
		if (!ext) {
		    alert("Sorry, floating point textures are unsupported.");
		    return;
		}
	}

	function initBuffers(gl, width, height){
		var buffer = gl.createBuffer();

		gl.bindBuffer(gl.ARRAY_BUFFER, buffer);

		var vertices = new Float32Array([1.0,  1.0,  0.0,
        -1.0,  1.0,  0.0,
         1.0, -1.0,  0.0,
        -1.0, -1.0,  0.0]);

        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
        this.buffer = buffer;

        // AmbientBuffer
        /*var depthTexture = createOutputR32FTexture(gl, width, height);
        var fb = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, depthTexture, 0);
        gl.drawBuffers([gl.COLOR_ATTACHMENT0]);

        this.depthTexture = depthTexture;
        this.ambientBuffer = fb;*/

        // Reflections Buffer
        var pointOffsetTexture = createOutputRGBA16FTexture(gl, width, height);
        var pointOffsetSwapTexture = createOutputRGBA16FTexture(gl, width, height);
        var reflectionTexture = createOutputRGBA32FTexture(gl, width, height);
        var reflectionSwapTexture = createOutputRGBA32FTexture(gl, width, height);
        var depthTexture = createOutputR32FTexture(gl, width, height);
        var materialTexture = createOutputR32FTexture(gl, width, height);
        var flatNormalTexture = createOutputRGBA16FTexture(gl, width, height);
        var uvMapTexture = createOutputTexure(gl, width, height);
        var reflection
        fb = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, pointOffsetTexture, 0);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, reflectionTexture, 0);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT2, gl.TEXTURE_2D, depthTexture, 0);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT3, gl.TEXTURE_2D, materialTexture, 0);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT4, gl.TEXTURE_2D, flatNormalTexture, 0);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT5, gl.TEXTURE_2D, uvMapTexture, 0)
        gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1, gl.COLOR_ATTACHMENT2, gl.COLOR_ATTACHMENT3, gl.COLOR_ATTACHMENT4, gl.COLOR_ATTACHMENT5]);

        this.pointOffsetTexture = pointOffsetTexture;
        this.pointOffsetSwapTexture = pointOffsetSwapTexture;
        this.reflectionTexture = reflectionTexture;
        this.reflectionSwapTexture = reflectionSwapTexture;
        this.depthTexture = depthTexture;
        this.materialTexture = materialTexture;
        this.flatNormalTexture = flatNormalTexture;
        this.uvMapTexture = uvMapTexture;
        this.reflectionBuffer = fb;

        // Texture Buffer
        var ambientTexture = createOutputTexure(gl, width, height);
        var diffuseTexture = createOutputTexure(gl, width, height);  
        var normalsTexture = createOutputRGBA16FTexture(gl, width, height);
        var specularTexture = createOutputTexure(gl, width, height);
        fb = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, diffuseTexture, 0);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, ambientTexture, 0);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT2, gl.TEXTURE_2D, normalsTexture, 0);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT3, gl.TEXTURE_2D, specularTexture, 0);
        gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1, gl.COLOR_ATTACHMENT2, gl.COLOR_ATTACHMENT3]);

        this.diffuseTexture = diffuseTexture;
        this.ambientTexture = ambientTexture;
        this.normalsTexture = normalsTexture;
        this.specularTexture = specularTexture;
        this.materialBuffer = fb;

        // Draw Buffer
        var fbTexture = createOutputTexure(gl, width, height, true);
        var swapTexture = createOutputTexure(gl, width, height, true);
        fb = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, fbTexture, 0);

        this.fbTexture = fbTexture;
        this.swapTexture = swapTexture;
        this.colourBuffer = fb;

        // Filter Buffer
        fbTexture = createOutputTexure(gl, width, height);
        fb = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, fbTexture, 0);

        this.fbFilterTexture = fbTexture;
        this.fbFilter = fb;
	}

	function createShader(gl, shaderScript, type) {
		var shader = gl.createShader(type);
		gl.shaderSource(shader, shaderScript);
		gl.compileShader(shader);

		if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
		  alert("Error building shader: " + gl.getShaderInfoLog(shader));
		  return null;
		}

		return shader;
	}

	function loadTexture(gl, image){
		var targetTexture = gl.createTexture();
		gl.bindTexture(gl.TEXTURE_2D, targetTexture);

		gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, image.width, image.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, image);
		gl.generateMipmap(gl.TEXTURE_2D);

		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST_MIPMAP_LINEAR);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

		return targetTexture;
	}

	function createOutputTexure(gl, width, height, isLinear){
		var targetTexture = gl.createTexture();
		gl.bindTexture(gl.TEXTURE_2D, targetTexture);

		gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, isLinear ? gl.LINEAR : gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, isLinear ? gl.LINEAR : gl.NEAREST);

		return targetTexture;
	}

	function createOutputRGBAUITexture(gl, width, height){
		var targetTexture = gl.createTexture();
		gl.bindTexture(gl.TEXTURE_2D, targetTexture);

		gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8UI, width, height, 0, gl.RGBA_INTEGER, gl.UNSIGNED_BYTE, null);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

		return targetTexture;
	}

	function createOutputR32FTexture(gl, width, height){
		var targetTexture = gl.createTexture();
		gl.bindTexture(gl.TEXTURE_2D, targetTexture);

		gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, width, height, 0, gl.RED, gl.FLOAT, null);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

		return targetTexture;
	}

	function createOutputRGBA16FTexture(gl, width, height){
		var targetTexture = gl.createTexture();
		gl.bindTexture(gl.TEXTURE_2D, targetTexture);

		gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, width, height, 0, gl.RGBA, gl.HALF_FLOAT, null);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

		return targetTexture;
	}

	function createOutputRGBA32FTexture(gl, width, height){
		var targetTexture = gl.createTexture();
		gl.bindTexture(gl.TEXTURE_2D, targetTexture);

		gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0, gl.RGBA, gl.FLOAT, null);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
		gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

		return targetTexture;
	}

	var createProgram = function (gl, name, virtCode, fragCode) {
        var p = gl.createProgram();
        gl.attachShader(p, createShader(gl, virtCode, gl.VERTEX_SHADER));
        gl.attachShader(p, createShader(gl, fragCode, gl.FRAGMENT_SHADER));
        gl.linkProgram(p);

        if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
        	console.log(gl.getProgramInfoLog(p));
            console.log("Could not initialize shaders for: " + name);
        }

        return p;
    };

    function addShapeToShader(gl, program, shapeName){
    	program[shapeName] = {
    		location : gl.getUniformLocation(program, shapeName + ".location"),
    		r : gl.getUniformLocation(program, shapeName + ".r"),
    		materialId : gl.getUniformLocation(program, shapeName + ".materialId"),
    		type : gl.getUniformLocation(program, shapeName + ".type")
    	};
    }

    function addShapesToShader(gl, program){
    	addShapeToShader(gl, program, "s1");
    	addShapeToShader(gl, program, "s2");
    	addShapeToShader(gl, program, "s3");
    }

	function initShaders(gl){
		var program = createProgram(gl, "Raytrace", vertexShader, calcFrameShader);
		gl.useProgram(program);
		gl.enableVertexAttribArray(0);

		program.mvMatrixUniform = gl.getUniformLocation(program, "uMVMatrix");
		program.seed = gl.getUniformLocation(program, "uSeed");
		program.light = {
			colour: gl.getUniformLocation(program, "light.colour"),
			location: gl.getUniformLocation(program, "light.location"),
			softFactor: gl.getUniformLocation(program, "light.softFactor"),
			castShadow: gl.getUniformLocation(program, "light.castShadow"),
			objType: gl.getUniformLocation(program, "light.type")
		}
		program.rayDepth = gl.getUniformLocation(program, "rayDepth");
		program.normals = gl.getUniformLocation(program, "normals");
		program.eyeOffset = gl.getUniformLocation(program, "eyeOffset");
		program.rays = gl.getUniformLocation(program, "sceneRays");
		program.diffuseColour = gl.getUniformLocation(program, "diffuseColour");
		program.specularColour = gl.getUniformLocation(program, "specularColour");
		addShapesToShader(gl, program);

		this.program = program;
		gl.uniform1i(this.program.rayDepth, 0);
		gl.uniform1i(this.program.normals, 1);
		gl.uniform1i(this.program.eyeOffset, 2);
		gl.uniform1i(this.program.rays, 3);
		gl.uniform1i(this.program.diffuseColour, 4);
		gl.uniform1i(this.program.specularColour, 5);

		program = createProgram(gl, "Denoise", filterVertShader, filterFragShader);
		gl.useProgram(program);
		gl.enableVertexAttribArray(0);

		program.currentScreen = gl.getUniformLocation(program, "uCurrentScreen");
		program.lastScreen = gl.getUniformLocation(program, "uLastScreen");

		this.filterProgram = program;
		gl.uniform1i(this.filterProgram.currentScreen, 6);
		gl.uniform1i(this.filterProgram.lastScreen, 7);

		program = createProgram(gl, "Rasterize", rasterVertShader, rasterFragShader);
		gl.useProgram(program);
		gl.enableVertexAttribArray(0);

		program.texture = gl.getUniformLocation(program, "uTexture");

		this.rasterizeProgram = program;
		gl.uniform1i(this.rasterizeProgram.texture, 7);

		program = createProgram(gl, "Reflection", vertexShader, calcReflectionsShader);
		gl.useProgram(program);
		gl.enableVertexAttribArray(0);

		program.mvMatrixUniform = gl.getUniformLocation(program, "uMVMatrix");
		program.seed = gl.getUniformLocation(program, "uSeed");
		program.currentOffset = gl.getUniformLocation(program, "currentOffset");
		program.currentRay = gl.getUniformLocation(program, "currentRay");
		program.normals = gl.getUniformLocation(program, "normals");
		program.bounce = gl.getUniformLocation(program, "bounce");
		program.specularColour = gl.getUniformLocation(program, "specularColour");
		addShapesToShader(gl, program);

		this.reflectionProgram = program;

		gl.uniform1i(this.reflectionProgram.currentOffset, 2);
		gl.uniform1i(this.reflectionProgram.normals, 1);
		gl.uniform1i(this.reflectionProgram.currentRay, 3);
		gl.uniform1i(this.reflectionProgram.specularColour, 5);

		program = createProgram(gl, "Texture", vertexShader, calculateMaterialShader);
		gl.useProgram(program);
		gl.enableVertexAttribArray(0);

		program.mvMatrixUniform = gl.getUniformLocation(program, "uMVMatrix");
		program.rayDepth = gl.getUniformLocation(program, "rayDepth");
		program.eyeOffset = gl.getUniformLocation(program, "eyeOffset");
		program.rays = gl.getUniformLocation(program, "sceneRays");
		program.sunAngle = gl.getUniformLocation(program, "sunAngle");
		program.flatNormals = gl.getUniformLocation(program, "flatNormals");
		program.materialId = gl.getUniformLocation(program, "materialId");
		program.uvMap = gl.getUniformLocation(program, "uvMap");
		program.material = {
			id: gl.getUniformLocation(program, "material.id"),
			diffuseColour: gl.getUniformLocation(program, "material.diffuseColour"),
			specularColour: gl.getUniformLocation(program, "material.specularColour"),
			roughness: gl.getUniformLocation(program, "material.roughness"),
			reflectionAmount: gl.getUniformLocation(program, "material.reflectionAmount"),
			diffuseTexture: gl.getUniformLocation(program, "material.diffuseTexture"),
			normalTexture: gl.getUniformLocation(program, "material.normalTexture"),
			roughnessTexture: gl.getUniformLocation(program, "material.roughnessTexture")
		};
		//addShapesToShader(gl, program);

		this.materialProgram = program;
		gl.uniform1i(this.materialProgram.rayDepth, 0);
		gl.uniform1i(this.materialProgram.eyeOffset, 2);
		gl.uniform1i(this.materialProgram.rays, 3);
		gl.uniform1i(this.materialProgram.material.diffuseTexture, 4);
		gl.uniform1i(this.materialProgram.material.normalTexture, 8);
		gl.uniform1i(this.materialProgram.material.roughnessTexture, 9);
		gl.uniform1i(this.materialProgram.flatNormals, 1);
		gl.uniform1i(this.materialProgram.materialId, 10);
		gl.uniform1i(this.materialProgram.uvMap, 11);
	}

	Renderer.prototype.init = function(canvas, width, height) {
		this.canvas = canvas;
		this.context = canvas.getContext("webgl2", { preserveDrawingBuffer: true });

		this.camera = new Camera(0, 0, -10.0, 0, 0, 0, width/height, 45.0);

		this.bounceMax = 2;

		initWebGL.call(this, this.context);
		initBuffers.call(this, this.context, width, height);
		initShaders.call(this, this.context);


	};

	var testDist = -10.0;
	var heartLocation = 0.0;
	var rad = 1.0;
	var heartBeat = [1.0, 0.75, 3.0, 0.25, 1.0];
	Renderer.prototype.update = function(timeDelta) {
		//mat4.translate(0, 0, timeDelta * -0.01, this.mvMatrix);

		// Calculate the radius as a pulse
		/*heartLocation += (timeDelta * 0.003);

		while(heartLocation >= heartBeat.length){
			heartLocation -= heartBeat.length;
		}

		var p1 = Math.floor(heartLocation);
		var p2 = Math.ceil(heartLocation);
		var dif = heartLocation - p1;

		if(p2 >= heartBeat.length){
			p2 = 0;
		}

		rad = heartBeat[p1] * (1.0 - dif) + heartBeat[p2] * dif;
		rad += 2.0;*/

		this.camera.update(timeDelta);
	};

	function swapBuffers(gl){
		// Swap the colourBuffer
		//gl.bindFramebuffer(gl.FRAMEBUFFER, this.colourBuffer);
		gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbFilter);

		/*sb = this.swapTexture;
		this.swapTexture = this.fbTexture;
		this.fbTexture = sb;*/

		sb = this.fbFilterTexture;
		this.fbFilterTexture = this.swapTexture;
		this.swapTexture = sb;

		gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.fbFilterTexture, 0);
	}

	function swapReflections(gl){
		// Swap the reflecitons buffer
		gl.bindFramebuffer(gl.FRAMEBUFFER, this.reflectionBuffer);

		var sb = this.pointOffsetSwapTexture;
		this.pointOffsetSwapTexture = this.pointOffsetTexture;
		this.pointOffsetTexture = sb;

		gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, sb, 0);

		sb = this.reflectionSwapTexture;
		this.reflectionSwapTexture = this.reflectionTexture;
		this.reflectionTexture = sb;

		gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, sb, 0);
	}

	function setShape(gl, shape, program, index){
		var s = program["s" + (index + 1)];

		gl.uniform3fv(s.location, shape.location);
		gl.uniform1f(s.r, shape.radius);
		gl.uniform1i(s.materialId, shape.materialId);
		gl.uniform1i(s.type, shape.type);
	}

	function setShapes(gl, environment, program){
		for(var i = 0; i < environment.objects.length; ++i){
			setShape(gl, environment.objects[i], program, i);
		}
	}

	function drawReflections(gl, environment, blendFactor, bounce){
		gl.bindFramebuffer(gl.FRAMEBUFFER, this.reflectionBuffer);
		gl.clear(gl.COLOR_BUFFER_BIT);
		gl.useProgram(this.reflectionProgram);

		setShapes(gl, environment, this.reflectionProgram);

		gl.uniformMatrix4fv(this.reflectionProgram.mvMatrixUniform, false, this.camera.mvMatrix);
		gl.uniform1f(this.reflectionProgram.seed, Math.random());
		gl.uniform1i(this.reflectionProgram.bounce, bounce);

		gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
		gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
		gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
	}

	function drawScene(gl, environment, blendFactor) {
		gl.enable(gl.BLEND);

		var lights = environment.lights;
		var light;

		gl.bindFramebuffer(gl.FRAMEBUFFER, this.colourBuffer);

		// Draw the ambient colour
		gl.useProgram(this.rasterizeProgram);

		gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
		gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
		gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

		// Draw Each Light
		gl.useProgram(this.program);

		setShapes(gl, environment, this.program);

		gl.uniformMatrix4fv(this.program.mvMatrixUniform, false, this.camera.mvMatrix);
		gl.uniform1f(this.program.seed, Math.random());

		for(var i = 0; i < lights.length; ++i){
			light = lights[i];

			gl.uniform3f(this.program.light.colour, light.r, light.g, light.b);
			gl.uniform3f(this.program.light.location, light.x, light.y, light.z);
			gl.uniform1f(this.program.light.softFactor, light.softFactor);
			gl.uniform1i(this.program.light.castShadow, light.castShadow ? 1 : 0);
			gl.uniform1i(this.program.light.objType, light.objType);

			gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
			gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
			gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
		}

		gl.disable(gl.BLEND);
	}

	function getTexture(gl, texture){
		if(texture == null || !texture.isLoaded()){
			return null;
		}

		if(texture.glImage == null){
			texture.glImage = loadTexture(gl, texture.image);
		}

		return texture.glImage;
	}

	function drawMaterial(gl, lightSun, environment, blendFactor){
		var materials = environment.materials;
		var material;

		gl.bindFramebuffer(gl.FRAMEBUFFER, this.materialBuffer);
		gl.clear(gl.COLOR_BUFFER_BIT);

		gl.useProgram(this.materialProgram);

		gl.uniformMatrix4fv(this.materialProgram.mvMatrixUniform, false, this.camera.mvMatrix);
		gl.uniform3f(this.materialProgram.sunAngle, lightSun.x, lightSun.y, lightSun.z);

		for(var i = 0; i < materials.length; ++i){
			material = materials[i];

			// Set the material data
			gl.uniform1i(this.materialProgram.material.id, material.id);
			gl.uniform3fv(this.materialProgram.material.diffuseColour, material.diffuseColour);
			gl.uniform3fv(this.materialProgram.material.specularColour, material.specularColour);
			gl.uniform1f(this.materialProgram.material.roughness, material.roughness);
			gl.uniform1f(this.materialProgram.material.reflectionAmount, material.reflectionAmount);

			gl.activeTexture(gl.TEXTURE4);
			gl.bindTexture(gl.TEXTURE_2D, getTexture(gl, material.diffuseTexture));

			gl.activeTexture(gl.TEXTURE8);
			gl.bindTexture(gl.TEXTURE_2D, getTexture(gl, material.normalTexture));

			gl.activeTexture(gl.TEXTURE9);
			gl.bindTexture(gl.TEXTURE_2D, getTexture(gl, material.roughnessTexture));

			gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
			gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
			gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
		}
		
	}

	Renderer.prototype.updateDraw = function(lightSun, environment, x, y, width, height, blendFactor = 1.0) {
		var gl = this.context;
		//gl.viewport(x, y, width, height);

		gl.bindFramebuffer(gl.FRAMEBUFFER, this.colourBuffer);
		gl.clear(gl.COLOR_BUFFER_BIT);

		// We are now setting the textures externally from the function to avoid setting the same texture multiple times.
		for(var bounce = 0; bounce < this.bounceMax; ++bounce){
			// Setup Reflections
			gl.activeTexture(gl.TEXTURE2);
			gl.bindTexture(gl.TEXTURE_2D, this.pointOffsetSwapTexture);

			//gl.activeTexture(gl.TEXTURE1);
			//gl.bindTexture(gl.TEXTURE_2D, this.normalsTexture); // Since we never use it at the beginning and set it later in this loop, there's no point in setting it here.

			gl.activeTexture(gl.TEXTURE3);
			gl.bindTexture(gl.TEXTURE_2D, this.reflectionSwapTexture);

			//gl.activeTexture(gl.TEXTURE5);
			//gl.bindTexture(gl.TEXTURE_2D, this.specularTexture); // Since we never use it at the beginning and set it later in this loop, there's no point in setting it here.

			drawReflections.call(this, gl, environment, blendFactor, bounce);

			// Draw Materials
			gl.activeTexture(gl.TEXTURE0);
			gl.bindTexture(gl.TEXTURE_2D, this.depthTexture);

			gl.activeTexture(gl.TEXTURE2);
			gl.bindTexture(gl.TEXTURE_2D, this.pointOffsetTexture);

			gl.activeTexture(gl.TEXTURE3);
			gl.bindTexture(gl.TEXTURE_2D, this.reflectionTexture);

			gl.activeTexture(gl.TEXTURE1);
			gl.bindTexture(gl.TEXTURE_2D, this.flatNormalTexture);

			gl.activeTexture(gl.TEXTURE10);
			gl.bindTexture(gl.TEXTURE_2D, this.materialTexture);

			gl.activeTexture(gl.TEXTURE11);
			gl.bindTexture(gl.TEXTURE_2D, this.uvMapTexture);
			drawMaterial.call(this, gl, lightSun, environment, blendFactor);

			// Draw the lights
			gl.activeTexture(gl.TEXTURE7);
			gl.bindTexture(gl.TEXTURE_2D, this.ambientTexture);

			gl.activeTexture(gl.TEXTURE1);
			gl.bindTexture(gl.TEXTURE_2D, this.normalsTexture);

			gl.activeTexture(gl.TEXTURE4);
			gl.bindTexture(gl.TEXTURE_2D, this.diffuseTexture);

			gl.activeTexture(gl.TEXTURE5);
			gl.bindTexture(gl.TEXTURE_2D, this.specularTexture);

			drawScene.call(this, gl, environment, blendFactor);

			// Swap buffers
			swapReflections.call(this, gl);
		}
	}

	Renderer.prototype.draw = function(x, y, width, height){
		var gl = this.context;

		swapBuffers.call(this, gl);

		// Perform Filter
		gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbFilter);
		//gl.viewport(x, y, width, height);

		gl.useProgram(this.filterProgram);

		gl.activeTexture(gl.TEXTURE6);
		gl.bindTexture(gl.TEXTURE_2D, this.swapTexture);

		gl.activeTexture(gl.TEXTURE7);
		gl.bindTexture(gl.TEXTURE_2D, this.fbTexture);

		gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
		gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
		gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
	}

	Renderer.prototype.rasterize = function(environment, x, y, width, height, texture){
		var gl = this.context;

		// Draw to screen
		gl.bindFramebuffer(gl.FRAMEBUFFER, null);
		gl.useProgram(this.rasterizeProgram);

		//gl.viewport(x, y, width, height);

		gl.activeTexture(gl.TEXTURE7);
		gl.bindTexture(gl.TEXTURE_2D, texture);

		gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
		gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
		gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
	};
}