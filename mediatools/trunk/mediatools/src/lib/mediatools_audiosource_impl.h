

#include <string>
#include <vector>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
}


class mediatools_audiosource_impl {
    public:
        int d_type;
        bool d_ready;
        std::string d_filename;

        mediatools_audiosource_impl();
        bool open(std::string);
        bool open_mpeg();
        void readData(std::vector<int16_t> &r);
    
        AVFormatContext* d_format_ctx;
        AVCodecContext* d_codec_ctx;
        AVCodec *d_codec;
        AVFrame *d_frame;
        AVPacket d_packet;
};
