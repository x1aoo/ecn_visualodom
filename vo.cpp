#include <ecn_visualodom/video_loop.h>
#include <ecn_visualodom/visual_odom.h>
#include <ecn_visualodom/ar_display.h>

using namespace std;


int main(int argc, char ** argv)
{
    // video source and image
    VideoLoop cap("../armen.mp4");
    cv::Mat im;
    cap.read(im);

    // camera parameters
    vpCameraParameters cam(684.169232, 680.105767, 365.163397, 292.358876);

    // initial pose: translation known, rotation to be estimated later
    vpHomogeneousMatrix cMo0(-0.1,-0.1,0.7,0,0,0);

    // visual odometer
    bool relative_to_initial = true;
    VisualOdom vo(cam, relative_to_initial);

    // give 3D hints
    // we look for a more or less horizontal plane
     vo.setNormalGuess(0,0,1);
    // we know the initial translation
    vo.setInitialTranslation(cMo0.getTranslationVector());

    // AR
    ModelDisplay ar(cam, im, true);

    // cMo = current pose, M = relative pose between two images (from previous to next)
    vpHomogeneousMatrix cMo, M;

    // loop variables
    vpRotationMatrix R;
    vpColVector nor;
    double d;
    bool compute_initial_pose = true;

    while(ar.continueRendering())
    {
        // get next image
        cap.read(im);

        // process and compute 2M1
        // returns True if succeded
        if(vo.process(im,M))
        {
            if(compute_initial_pose)
            {
                // initialize absolute pose from normal
                vo.getRotationFromNormal(R);
                // update initial pose
                cMo0.insert(R);

                // write current estimation
                cMo = cMo0;

                compute_initial_pose = false;
            }


            // update cMo
             cMo = M*cMo0;

        }

        // AR or frame display
        ar.display(im, cMo);

        // refresh display
        cv::waitKey(100);

    }












}
