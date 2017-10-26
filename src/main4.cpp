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

int wsx=600, wsy = 600;
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
	void setup()
	{
		_controlfp(_DN_FLUSH, _MCW_DN);

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
	float noiseProgressSpeed;
	
	void draw()
	{
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

		/*Sleep(50);*/
		sw::endFrame();
		cfg1::print();
		my_console::endFrame();

		if(pause)
			Sleep(50);
		//Sleep(2000);
	}
	// from the matrix.rotate method with hardcoded axis (1, 1, 1)
	void rotateHue_ip( Vec3f& v, float angle )
	{
		typedef float T;
		T sina = math<T>::sin(angle);
		T cosa1 = 1.0f - math<T>::cos(angle);
		T m = cosa1 - sina;
		T p = cosa1 + sina;

		T rx = v.x + m * v.y + p * v.z;
		T ry = p * v.x + v.y + m * v.z;
		T rz = m * v.x + p * v.y + v.z;

		v.x = rx;
		v.y = ry;
		v.z = rz;
	}
	void updateIt() {
		if(!pause) {
			int ksize = 9;
			float sigma = sigmaFromKsize(8);
			img = separableConvolve<Vec3f, WrapModes::GetWrapped>(img, getGaussianKernel(ksize, sigma));
			Array2D<float> imgGrayscale(img.Size(), 0.0f);
			float maxLen = Vec3f::one().length();
			forxy(imgGrayscale) {
				imgGrayscale(p) = img(p).length()/maxLen;
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
					img(p) += -dot * 3.8 * complexToColor_HSV(curvDirs(p));
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
		auto sorted = a.clone();
		auto beginIt = (float*)(sorted.begin());
		auto endIt = (float*)(sorted.end())+2;
		std::sort(beginIt, endIt);
		auto minn = sorted.data[int(sorted.area*.1)];
		auto maxx = sorted.data[int(sorted.area*.9)];
		auto b = a.clone();
		forxy(b) {
			b(p) -= Vec3f::one() * minn;
			b(p) /= Vec3f::one() * (maxx - minn);
		}
		return b;
	}
	void renderIt() {
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
int WINAPI WinMain(HINSTANCE, HINSTANCE, LPSTR, int) {
	return mainFuncImpl(new SApp());
}

