#include "precompiled.h"
#if 1
#include "util.h"
#include "stuff.h"
#include "shade.h"
#include "gpgpu.h"
#include "gpuBlur2_4.h"
#include "stefanfw.h"

#include "hdrwrite.h"
#include "simplexnoise.h"

#include "colorspaces.h"
#include "easyfft.h"

#define GLSL(sh) #sh

int wsx=600, wsy = 600;
int scale = 2;
int sx = wsx / ::scale;
int sy = wsy / ::scale;

Array2D<vec3> img(sx, sy);

bool pause;


vec3 complexToColor_HSV(vec2 comp) {
	float hue = (float)M_PI+(float)atan2(comp.y,comp.x);
	hue /= (float)(2*M_PI);
	hue = fmod(hue + .2f, 1.0f);

	HslF hsl(hue, 1.0f, .5);
	auto rgb = FromHSL(hsl);
	rgb /= dot(rgb, vec3(1.0) / 3.0f);
	return rgb;
}

void updateConfig() {
}

struct SApp : App {
	void setup()
	{
		enableDenormalFlushToZero();

		createConsole();
		disableGLReadClamp();
		stefanfw::eventHandler.subscribeToEvents(*this);
		setWindowSize(wsx, wsy);

		vec2 center = vec2(img.Size()) / 2.0f;
		forxy(img) {
			img(p) = vec3(ci::randFloat(), ci::randFloat(), ci::randFloat()) * (distance(vec2(p), center) < sy / 3 ? 1.0f : 0.0f);
		}
	}
	void update()
	{
		stefanfw::beginFrame();
		stefanUpdate();
		stefanDraw();
		stefanfw::endFrame();
	}
	void keyDown(KeyEvent e)
	{
		if(keys['p'] || keys['2'])
		{
			pause = !pause;
		}
	}
	float noiseProgressSpeed;
	
	void stefanUpdate() {
		if(!pause) {
			int ksize = 9;
			float sigma = sigmaFromKsize(8);
			img = separableConvolve<vec3, WrapModes::GetWrapped>(img, getGaussianKernel(ksize, sigma));
			Array2D<float> imgGrayscale(img.Size());
			float maxLen = length(vec3(1.0f));
			forxy(imgGrayscale) {
				imgGrayscale(p) = length(img(p))/maxLen;
			}
			Array2D<vec2> curvDirs(img.Size());
			auto grads = ::get_gradients(imgGrayscale);
			for(int x = 0; x < sx; x++)
			{
				for(int y = 0; y < sy; y++)
				{
					vec2 p = vec2(x,y);
					vec2 grad = safeNormalized(grads(x, y));
					grad = vec2(-grad.y, grad.x);;
					vec2 grad_a = safeNormalized(getBilinear(grads, p+grad));
					grad_a = -vec2(-grad_a.y, grad_a.x);
					vec2 grad_b = safeNormalized(getBilinear(grads, p-grad));
					grad_b = vec2(-grad_b.y, grad_b.x);
					vec2 dir = grad_a + grad_b;
					curvDirs(x, y) = dir;
				}
			}
			forxy(img) {
				float dot = glm::dot(curvDirs(p), grads(p));
				if(dot < 0) {
					img(p) += -dot * 3.8f * complexToColor_HSV(curvDirs(p));
				}
			}
			img = to01(img);
			
			if(mouseDown_[0])
			{
				vec2 scaledm = vec2(mouseX * (float)sx, mouseY * (float)sy);
				Area a(scaledm, scaledm);
				int r = 10;
				a.expand(r, r);
				for(int x = a.x1; x <= a.x2; x++)
				{
					for(int y = a.y1; y <= a.y2; y++)
					{
						vec2 v = vec2(x, y) - scaledm;
						float w = max(0.0f, 1.0f - length(v) / r);
						w=max(0.0f,w);
						w = 3 * w * w - 2 * w * w * w;
						img.wr(x, y) = lerp(img.wr(x, y), vec3(1.0f), w);
					}
				}
			}
		}

		if (pause)
			Sleep(50);
	}
	inline Array2D<vec3> to01(Array2D<vec3> a) {
		auto sorted = a.clone();
		auto beginIt = (float*)(sorted.begin());
		auto endIt = (float*)(sorted.end())+2;
		std::sort(beginIt, endIt);
		auto minn = sorted.data[int(sorted.area*.1)];
		auto maxx = sorted.data[int(sorted.area*.9)];
		auto b = a.clone();
		forxy(b) {
			b(p) -= vec3(1.0f) * minn;
			b(p) /= vec3(1.0f) * (maxx - minn);
		}
		return b;
	}
	void stefanDraw() {
		gl::clear(Color(0, 0, 0));

		gl::setMatricesWindow(getWindowSize(), false);

		auto tex = gtex(img);

		auto texb = gpuBlur2_4::run(tex, 1);

		tex = shade2(tex, texb,
			"vec3 f = fetch3();"
			"vec3 fb = fetch3(tex2);"
			"vec3 diff = f - fb;"
			"diff = vec3(dot(diff, vec3(1.0/3.0)));"
			"_out = f + diff * 3.0;"
			);

		tex = shade2(tex,
			"vec3 c = fetch3();"
			"c *= .4;"
			"_out = c;"
			);

		gl::draw(tex, getWindowBounds());
	}
#endif
};

		
CINDER_APP(SApp, RendererGl)

