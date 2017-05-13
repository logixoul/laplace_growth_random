#include "precompiled.h"
#if 1
#include "util.h"
#include "stuff.h"
#include "shade.h"
#include "gpgpu.h"
#include "gpuBlur2_4.h"
#include "cfg1.h"
#include "sw.h"
#include "my_console.h"
#include "hdrwrite.h"
#include <float.h>
#include "simplexnoise.h"
#include "mainfunc_impl.h"
#include "colorspaces.h"
#include "easyfft.h"

#define GLSL(sh) #sh

int wsx=800, wsy = 800 * (800.0f / 1280.0f);
int scale = 2;
int sx = wsx / scale;
int sy = wsy / scale;
bool mouseDown_[3];
bool keys[256];
gl::Texture::Format gtexfmt;
float noiseTimeDim = 0.0f;
float heightmapTimeDim = 0.0f;
const int MAX_AGE = 100;
gl::Texture texToDraw;
bool texOverride = false;

Array2D<Vec3f> img(sx, sy);

float mouseX, mouseY;
bool pause;
bool keys2[256];

Vec3f complexToColor_HSV(Vec2f comp) {
	float hue = (float)M_PI+(float)atan2(comp.y,comp.x);
	hue /= (float)(2*M_PI);
	hue = fmod(hue + .2f, 1.0f);
	//float lightness = comp.length();
	//cout << complex << " --> " << color << endl;

	//lightness /= lightness + 1.0f;
	HslF hsl(hue, 1.0f, .5);
	auto rgb = FromHSL(hsl);
	rgb /= rgb.dot(Vec3f::one() / 3.0f);
	return rgb;
}

void updateConfig() {
}

struct SApp : AppBasic {
	Rectf area;
		
	void setup()
	{
		//keys2['0']=keys2['1']=keys2['2']=keys2['3']=true;
		//_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

		_controlfp(_DN_FLUSH, _MCW_DN);

		area = Rectf(0, 0, (float)sx-1, (float)sy-1).inflated(Vec2f::zero());

		glClampColor(GL_CLAMP_FRAGMENT_COLOR, GL_FALSE);
		glClampColor(GL_CLAMP_READ_COLOR, GL_FALSE);
		glClampColor(GL_CLAMP_VERTEX_COLOR, GL_FALSE);
		gtexfmt.setInternalFormat(hdrFormat);
		setWindowSize(wsx, wsy);

		glEnable(GL_POINT_SMOOTH);

		Vec2f center = Vec2f(img.Size()) / 2.0f;
		forxy(img) {
			img(p) = Vec3f::one() * (p.distance(center) < sy / 3 ? 1 : 0);
		}
	}
	void keyDown(KeyEvent e)
	{
		keys[e.getChar()] = true;
		if(e.isControlDown()&&e.getCode()!=KeyEvent::KEY_LCTRL)
		{
			keys2[e.getChar()] = !keys2[e.getChar()];
			return;
		}
		if(keys['r'])
		{
		}
		if(keys['p'] || keys['2'])
		{
			pause = !pause;
		}
	}
	void keyUp(KeyEvent e)
	{
		keys[e.getChar()] = false;
	}
	
	void mouseDown(MouseEvent e)
	{
		mouseDown_[e.isLeft() ? 0 : e.isMiddle() ? 1 : 2] = true;
	}
	void mouseUp(MouseEvent e)
	{
		mouseDown_[e.isLeft() ? 0 : e.isMiddle() ? 1 : 2] = false;
	}
	Vec2f direction;
	Vec2f lastm;
	void mouseDrag(MouseEvent e)
	{
		mm();
	}
	void mouseMove(MouseEvent e)
	{
		mm();
	}
	void mm()
	{
		direction = getMousePos() - lastm;
		lastm = getMousePos();
	}
	Vec2f reflect(Vec2f const & I, Vec2f const & N)
	{
		return I - N * N.dot(I) * 2.0f;
	}
	float noiseProgressSpeed;
	
	void draw()
	{
		::texOverride = false;

		my_console::beginFrame();
		sw::beginFrame();
		static bool first = true;
		first = false;

		wsx = getWindowSize().x;
		wsy = getWindowSize().y;

		mouseX = getMousePos().x / (float)wsx;
		mouseY = getMousePos().y / (float)wsy;
		/*noiseProgressSpeed=cfg1::getOpt("noiseProgressSpeed", .00008f,
			[&]() { return keys['s']; },
			[&]() { return expRange(mouseY, 0.01f, 100.0f); });*/
		
		gl::clear(Color(0, 0, 0));

		updateIt();
		
		renderIt();

		/*float hue = 0;
		float lightness = .5;
		HslF hsl(hue, 1.0f, lightness);
 		cout << "rgb=" << FromHSL(hsl) << endl;*/

		/*Sleep(50);*/my_console::clr();
		sw::endFrame();
		cfg1::print();
		my_console::endFrame();

		if(pause)
			Sleep(50);
		//Sleep(2000);
	}
	void updateIt() {
		if(!pause) {
			int ksize = 9;
			float sigma = sigmaFromKsize(8);
			img = separableConvolve<Vec3f, WrapModes::GetWrapped>(img, getGaussianKernel(ksize, sigma));
			Array2D<float> imgGrayscale(img.Size(), 0.0f);
			forxy(imgGrayscale) {
				imgGrayscale(p) = img(p).dot(Vec3f::one() / 3.0f);
			}
			Array2D<Vec2f> curvDirs(img.Size(), Vec2f::zero());
			auto grads = ::get_gradients(imgGrayscale);
			for(int x = 0; x < sx; x++)
			{
				for(int y = 0; y < sy; y++)
				{
					Vec2f p = Vec2f(x,y);
					Vec2f grad = grads(x, y).safeNormalized();
					grad = Vec2f(-grad.y, grad.x);;
					Vec2f grad_a = getBilinear(grads, p+grad).safeNormalized();
					grad_a = -Vec2f(-grad_a.y, grad_a.x);
					Vec2f grad_b = getBilinear(grads, p-grad).safeNormalized();
					grad_b = Vec2f(-grad_b.y, grad_b.x);
					Vec2f dir = grad_a + grad_b;
					curvDirs(x, y) = dir;
				}
			}
			forxy(img) {
				float dot = curvDirs(p).dot(grads(p));
				if(dot < 0) {
					img(p) += -dot * 4.0 * complexToColor_HSV(curvDirs(p));
					//if(c != Vec3f::one())
					//aaPoint(img, Vec2f(p) + curvDirs(p).safeNormalized(), dot * 10.0f);
				}
			}
			img = to01(img);
			
			if(mouseDown_[0])
			{
				Vec2f scaledm = Vec2f(mouseX * (float)sx, mouseY * (float)sy);
				Area a(scaledm, scaledm);
				int r = 10;
				a.expand(r, r);
				for(int x = a.x1; x <= a.x2; x++)
				{
					for(int y = a.y1; y <= a.y2; y++)
					{
						Vec2f v = Vec2f(x, y) - scaledm;
						float w = max(0.0f, 1.0f - v.length() / r);
						w=max(0.0f,w);
						w = 3 * w * w - 2 * w * w * w;
						img.wr(x, y) = lerp(img.wr(x, y), Vec3f::one(), w);
					}
				}
			}
		}
	}
	inline Array2D<Vec3f> to01(Array2D<Vec3f> a) {
		auto beginIt = (float*)(a.begin());
		auto endIt = (float*)(a.end())+2;
		auto minn = *std::min_element(beginIt, endIt);
		auto maxx = *std::max_element(beginIt, endIt);
		auto b = a.clone();
		forxy(b) {
			b(p) -= Vec3f::one() * minn;
			b(p) /= Vec3f::one() * (maxx - minn);
		}
		return b;
	}
	void renderIt() {
		auto tex = gtex(img);
		
		auto texb = gpuBlur2_4::run(tex, 6);

		//static auto gradientMap = gl::Texture(ci::loadImage("gradient.png"));

		auto texAdapted = shade2(tex, texb,
			"vec3 f = fetch3();"
			"vec3 fb = fetch3(tex2);"
			"_out = .5 * f / fb;");
		
		tex = shade2(texAdapted,
			"vec3 f = fetch3();"
			/*"float L = getL(f);"*
			"f /= L + 1.0;"*/
			"f /= f + vec3(1.0);"
			"f *= 1.3;"
			//"vec3 gradientMapC = fetch3(tex2, vec2(f, 0));"
			//"vec3 c = gradientMapC;"
			"vec3 c = vec3(f);"
			"_out = c;",
			ShadeOpts(),
			FileCache::get("stuff.fs")
			);

		texb = gpuBlur2_4::run(tex, 1);

		tex = shade2(tex, texb,
			"vec3 f = fetch3();"
			"vec3 fb = fetch3(tex2);"
			"vec3 diff = f - fb;"
			"diff = vec3(dot(diff, vec3(1.0/3.0)));"
			"_out = f + diff * 2.0;"
			);

		/*tex = shade2(tex,
			"vec3 f = fetch3();"
			"f /= pow(dot(f, vec3(1.0/3.0)), .3);"
			"f *= .7;"
			"vec3 grad = g();"
			"f *= .5 + max(0.0, dot(grad, H));"
			"f += getSpecular(grad);"
			//"f = smoothstep(.0, 1., f);"
			"_out = f;"
			, ShadeOpts(),
			GLSL(
				// https://www.shadertoy.com/view/4ttXzj
				vec3 ld = normalize(vec3(1.0, 2.0, 3.));
				vec3 v = normalize(vec3(tc, 1.));
				vec3 H = normalize(ld + v);
				float fbm(vec2 uv) {
					vec3 c = fetch3(tex, uv);
					return dot(c, vec3(1.0/3.0)) * .09;
				}
				vec3 g()
				{
					vec2 off = vec2(0.0, .03);
					float t = fbm(tc);
					float x = t - fbm(tc + vec2(tsize.x, 0));
					float y = t - fbm(tc + vec2(tsize.y, 0));
					vec3 xv = vec3(tsize.x, x, 0);
					vec3 yv = vec3(0, y, tsize.y);
					return normalize(cross(xv, -yv)).xzy;
				}
				vec3 getSpecular(vec3 grad) {
					float S = max(0., dot(grad, H));
					S = pow(S, 16.0);
					return S * vec3(.4);
				}
			)
			);*/
		

		/*auto texForB = shade2(tex, texAdapted,
			"vec3 c = fetch3();"
			"float fAdapted = fetch1(tex2);"
			"float lum = pow(fAdapted+1.0, 3.0)-1.0;"
			"_out = c * lum;"
			);*/

		/*texb = gpuBlur2_4::run(tex, 4);

		tex = shade2(tex, texb,
			"vec3 f = fetch3();"
			"vec3 fb = fetch3(tex2);"
			"vec3 c = (f + fb) * .5;"
			//"c = pow(c + vec3(1.0), vec3(1.0/5.0)) - vec3(1.0);"
			"_out = c;"
			);*/

		gl::draw(tex, getWindowBounds());
	}
#endif
};
int WINAPI WinMain(HINSTANCE, HINSTANCE, LPSTR, int) {
	return mainFuncImpl(new SApp());
}

