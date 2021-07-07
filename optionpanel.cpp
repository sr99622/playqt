#include "optionpanel.h"
#include "mainwindow.h"
#include "Ffplay/cmdutils.h"

enum show_muxdemuxers {
    SHOW_DEFAULT,
    SHOW_DEMUXERS,
    SHOW_MUXERS,
};

OptionDialog::OptionDialog(QMainWindow *parent) : PanelDialog(parent)
{
    mainWindow = parent;
    setWindowTitle("Options");
    panel = new OptionPanel(mainWindow);
    QVBoxLayout *layout = new QVBoxLayout;
    layout->addWidget(panel);
    setLayout(layout);
}

int OptionDialog::getDefaultWidth()
{
    return defaultWidth;
}

int OptionDialog::getDefaultHeight()
{
    return defaultHeight;
}

OptionPanel::OptionPanel(QMainWindow *parent) : QWidget(parent)
{
    mainWindow = parent;

    currentOption = new QLabel();
    QPushButton *clear = new QPushButton("clear");
    QPushButton *close = new QPushButton("close");
    QPushButton *decoders = new QPushButton("decoders");
    QPushButton *decoder_set = new QPushButton("...");
    decoder_set->setMaximumWidth(30);
    QPushButton *filters = new QPushButton("filters");
    QPushButton *filter_set = new QPushButton("...");
    filter_set->setMaximumWidth(30);
    QPushButton *bsfs = new QPushButton("bsfs");
    QPushButton *pix_fmts = new QPushButton("pix_fmts");
    QPushButton *formats = new QPushButton("formats");
    QPushButton *sample_fmts = new QPushButton("sample_fmts");
    QPushButton *protocols = new QPushButton("protocols");
    QPushButton *layouts = new QPushButton("layouts");
    QPushButton *colors = new QPushButton("colors");
    QPushButton *demuxers = new QPushButton("demuxers");
    QPushButton *devices = new QPushButton("devices");
    QPushButton *help = new QPushButton("help");
    QPushButton *details = new QPushButton("details");
    QPushButton *config = new QPushButton("config");

    QLabel *lbl00 = new QLabel("| grep");
    filterEdit = new QLineEdit();
    filterEdit->setStyleSheet("QLineEdit {font-weight: bold;}");
    helpDisplay = new QTextEdit();
    helpDisplay->setWordWrapMode(QTextOption::NoWrap);
    helpDisplay->setFontFamily("Courier");
    helpDisplay->setFontWeight(QFont::Bold);

    QGridLayout *layout = new QGridLayout;
    layout->setContentsMargins(0, 0, 0, 0);
    layout->addWidget(lbl00,          0,  2,  1,  1, Qt::AlignLeft);
    layout->addWidget(filterEdit,     0,  3,  1,  5, Qt::AlignLeft);
    layout->addWidget(helpDisplay,    1,  2,  14, 6);
    layout->addWidget(currentOption,  0,  1,  1,  1, Qt::AlignLeft);
    layout->addWidget(decoder_set,    1,  0,  1,  1, Qt::AlignCenter);
    layout->addWidget(decoders,       1,  1,  1,  1, Qt::AlignCenter);
    layout->addWidget(filter_set,     2,  0,  1,  1, Qt::AlignCenter);
    layout->addWidget(filters,        2,  1,  1,  1, Qt::AlignCenter);
    layout->addWidget(bsfs,           3,  1,  1,  1, Qt::AlignCenter);
    layout->addWidget(pix_fmts,       4,  1,  1,  1, Qt::AlignCenter);
    layout->addWidget(formats,        5,  1,  1,  1, Qt::AlignCenter);
    layout->addWidget(sample_fmts,    6,  1,  1,  1, Qt::AlignCenter);
    layout->addWidget(protocols,      7,  1,  1,  1, Qt::AlignCenter);
    layout->addWidget(layouts,        8,  1,  1,  1, Qt::AlignCenter);
    layout->addWidget(colors,         9,  1,  1,  1, Qt::AlignCenter);
    layout->addWidget(demuxers,       10, 1,  1,  1, Qt::AlignCenter);
    layout->addWidget(devices,        11, 1,  1,  1, Qt::AlignCenter);
    layout->addWidget(help,           12, 1,  1,  1, Qt::AlignCenter);
    layout->addWidget(details,        13, 1,  1,  1, Qt::AlignCenter);
    layout->addWidget(config,         14, 1,  1,  1, Qt::AlignCenter);
    layout->addWidget(clear,          15, 3,  1,  1, Qt::AlignCenter);
    layout->addWidget(close,          15, 5,  1,  1, Qt::AlignCenter);

    //layout->setColumnStretch(0, 1);
    //layout->setColumnStretch(1, 1);
    layout->setColumnStretch(2, 10);
    layout->setColumnStretch(3, 10);

    setLayout(layout);

    connect(clear, SIGNAL(clicked()), this, SLOT(clear()));
    connect(close, SIGNAL(clicked()), this, SLOT(close()));
    connect(filterEdit, SIGNAL(textChanged(const QString&)), this, SLOT(filterChanged(const QString&)));
    connect(decoders, SIGNAL(clicked()), this, SLOT(decoders()));
    connect(filters, SIGNAL(clicked()), this, SLOT(filters()));
    connect(bsfs, SIGNAL(clicked()), this, SLOT(bsfs()));
    connect(pix_fmts, SIGNAL(clicked()), this, SLOT(pix_fmts()));
    connect(formats, SIGNAL(clicked()), this, SLOT(formats()));
    connect(sample_fmts, SIGNAL(clicked()), this, SLOT(sample_fmts()));
    connect(protocols, SIGNAL(clicked()), this, SLOT(protocols()));
    connect(layouts, SIGNAL(clicked()), this, SLOT(layouts()));
    connect(colors, SIGNAL(clicked()), this, SLOT(colors()));
    connect(demuxers, SIGNAL(clicked()), this, SLOT(demuxers()));
    connect(devices, SIGNAL(clicked()), this, SLOT(devices()));
    connect(help, SIGNAL(clicked()), this, SLOT(help()));
    connect(details, SIGNAL(clicked()), this, SLOT(details()));
    connect(config, SIGNAL(clicked()), this, SLOT(config()));
    connect(filter_set, SIGNAL(clicked()), this, SLOT(showParameterDialog()));

    parameterDialog = new ParameterDialog(mainWindow);
}

void OptionPanel::showConfig(const QString& str)
{
    helpDisplay->setText(helpDisplay->toPlainText() + str);
}

void OptionPanel::details()
{
    QMessageBox::StandardButton result = QMessageBox::question(MW->optionDialog, "FFMPEG details", "This utility will take a while to complete, during which time the application will become non-responsive.  Would you like to continue?");
    if (result == QMessageBox::Yes)
        show_help_default(NULL, NULL);
}

void OptionPanel::config()
{
    show_program_configs();
    MW->optionDialog->raise();
}

void OptionPanel::help()
{
    QString str;
    QTextStream(&str) << show_help_options(MW->co->options, "Main options:", 0, OPT_EXPERT, 0);
    QTextStream(&str) << show_help_options(MW->co->options, "Advanced options:", OPT_EXPERT, 0, 0);
    display(str, "Help");
}

void OptionPanel::devices()
{
    display(show_devices(), "Devices");
}

void OptionPanel::demuxers()
{
    display(show_demuxers(), "Demuxers");
}

void OptionPanel::colors()
{
    display(show_colors(), "Colors");
}

void OptionPanel::layouts()
{
    display(show_layouts(), "Layouts");
}

void OptionPanel::protocols()
{
    display(show_protocols(), "Protocols");
}

void OptionPanel::sample_fmts()
{
    display(show_sample_fmts(), "Sample Formats");
}

void OptionPanel::pix_fmts()
{
    display(show_pix_fmts(), "Pixel Formats");
}

void OptionPanel::formats()
{
    display(show_formats(), "Formats");
}

void OptionPanel::bsfs()
{
    display(show_bsfs(), "Bitstream Filters");
}

void OptionPanel::filters()
{
    display(show_filters(), "Filters");
}

void OptionPanel::decoders()
{
    display(print_codecs(0), "Decoders");
}

void OptionPanel::clear()
{
    currentOption->setText("");
    filterEdit->setText("");
    helpDisplay->setText("");
}

void OptionPanel::close()
{
    MW->optionDialog->hide();
}

void OptionPanel::test()
{
    QString str = print_codecs(0);
    optionList = str.split("\n");
    cout << "filterOptionList" << endl;
    helpDisplay->setText(filterOptionList(filterEdit->text()));
}

void OptionPanel::display(const QString& str, const QString& title)
{
    optionList = str.split("\n");
    helpDisplay->setText(filterOptionList(filterEdit->text()));
    currentOption->setText(title);
}

void OptionPanel::filterChanged(const QString& str)
{
    cout << str.toStdString() << endl;
    helpDisplay->setText(filterOptionList(str));
}

const QString OptionPanel::getOptionList()
{
    QString str;
    for (int i = 0; i < optionList.size(); i++) {
        QTextStream(&str) << optionList[i] << "\n";
    }
    return str;
}

const QString OptionPanel::filterOptionList(const QString& filter)
{
    QString str;
    //int counter = 0;
    for (int i = 0; i < optionList.size(); i++) {
        if (optionList[i].contains(filter)) {
            QTextStream(&str) << optionList[i] << "\n";
            //counter++;
        }
    }
    //cout << "counter: " << counter << endl;
    return str;
}

const QString OptionPanel::show_devices()
{
    return show_formats_devices(0, SHOW_DEFAULT);
}

const QString OptionPanel::show_demuxers()
{
    return show_formats_devices(0, SHOW_DEMUXERS);
}

void OptionPanel::show_muxers()
{
    show_formats_devices(0, SHOW_MUXERS);
}

const QString OptionPanel::show_formats()
{
    return show_formats_devices(0, SHOW_DEFAULT);
}

int OptionPanel::is_device(const AVClass *avclass)
{
    if (!avclass)
        return 0;
    return AV_IS_INPUT_DEVICE(avclass->category) || AV_IS_OUTPUT_DEVICE(avclass->category);
}

const QString OptionPanel::show_formats_devices(int device_only, int muxdemuxers)
{
    void *ifmt_opaque = NULL;
    const AVInputFormat *ifmt  = NULL;
    void *ofmt_opaque = NULL;
    const AVOutputFormat *ofmt = NULL;
    const char *last_name;
    int is_dev;

    char buffer[256];
    QString str;

    sprintf(buffer, "%s\n"
                    " D. = Demuxing supported\n"
                    " .E = Muxing supported\n"
                    " --\n", device_only ? "Devices:" : "File formats:");
    QTextStream(&str) << buffer;

    last_name = "000";
    for (;;) {
        int decode = 0;
        int encode = 0;
        const char *name      = NULL;
        const char *long_name = NULL;

        if (muxdemuxers !=SHOW_DEMUXERS) {
            ofmt_opaque = NULL;
            while ((ofmt = av_muxer_iterate(&ofmt_opaque))) {
                is_dev = is_device(ofmt->priv_class);
                if (!is_dev && device_only)
                    continue;
                if ((!name || strcmp(ofmt->name, name) < 0) &&
                    strcmp(ofmt->name, last_name) > 0) {
                    name      = ofmt->name;
                    long_name = ofmt->long_name;
                    encode    = 1;
                }
            }
        }
        if (muxdemuxers != SHOW_MUXERS) {
            ifmt_opaque = NULL;
            while ((ifmt = av_demuxer_iterate(&ifmt_opaque))) {
                is_dev = is_device(ifmt->priv_class);
                if (!is_dev && device_only)
                    continue;
                if ((!name || strcmp(ifmt->name, name) < 0) &&
                    strcmp(ifmt->name, last_name) > 0) {
                    name      = ifmt->name;
                    long_name = ifmt->long_name;
                    encode    = 0;
                }
                if (name && strcmp(ifmt->name, name) == 0)
                    decode = 1;
            }
        }
        if (!name)
            break;
        last_name = name;

        sprintf(buffer, " %s%s %-15s %s\n",
               decode ? "D" : " ",
               encode ? "E" : " ",
               name,
            long_name ? long_name:" ");
        QTextStream(&str) << buffer;
    }
    return str;
}

#define PRINT_CODEC_SUPPORTED(codec, field, type, list_name, term, get_name) \
    if (codec->field) {                                                      \
        const type *p = codec->field;                                        \
                                                                             \
        QTextStream(&str) << "    Supported " list_name ":";                 \
        while (*p != term) {                                                 \
            get_name(*p);                                                    \
            QTextStream(&str) << " " << name;                                \
            p++;                                                             \
        }                                                                    \
        QTextStream(&str) << "\n";                                           \
    }                                                                        \

const QString OptionPanel::print_codec(const AVCodec *c)
{
    int encoder = av_codec_is_encoder(c);

    char buffer[255];
    QString str;

    sprintf(buffer, "%s %s [%s]:\n", encoder ? "Encoder" : "Decoder", c->name,
           c->long_name ? c->long_name : "");
    QTextStream(&str) << buffer;

    QTextStream(&str) << "    General capabilities: ";
    if (c->capabilities & AV_CODEC_CAP_DRAW_HORIZ_BAND)
        QTextStream(&str) << "horizband ";
    if (c->capabilities & AV_CODEC_CAP_DR1)
        QTextStream(&str) << "dr1 ";
    if (c->capabilities & AV_CODEC_CAP_TRUNCATED)
        QTextStream(&str) << "trunc ";
    if (c->capabilities & AV_CODEC_CAP_DELAY)
        QTextStream(&str) << "delay ";
    if (c->capabilities & AV_CODEC_CAP_SMALL_LAST_FRAME)
        QTextStream(&str) << "small ";
    if (c->capabilities & AV_CODEC_CAP_SUBFRAMES)
        QTextStream(&str) << "subframes ";
    if (c->capabilities & AV_CODEC_CAP_EXPERIMENTAL)
        QTextStream(&str) << "exp ";
    if (c->capabilities & AV_CODEC_CAP_CHANNEL_CONF)
        QTextStream(&str) << "chconf ";
    if (c->capabilities & AV_CODEC_CAP_PARAM_CHANGE)
        QTextStream(&str) << "paramchange ";
    if (c->capabilities & AV_CODEC_CAP_VARIABLE_FRAME_SIZE)
        QTextStream(&str) << "variable ";
    if (c->capabilities & (AV_CODEC_CAP_FRAME_THREADS |
                           AV_CODEC_CAP_SLICE_THREADS |
                           AV_CODEC_CAP_AUTO_THREADS))
        QTextStream(&str) << "threads ";
    if (c->capabilities & AV_CODEC_CAP_AVOID_PROBING)
        QTextStream(&str) << "avoidprobe ";
    if (c->capabilities & AV_CODEC_CAP_INTRA_ONLY)
        QTextStream(&str) << "intraonly ";
    if (c->capabilities & AV_CODEC_CAP_LOSSLESS)
        QTextStream(&str) << "lossless ";
    if (c->capabilities & AV_CODEC_CAP_HARDWARE)
        QTextStream(&str) << "hardware ";
    if (c->capabilities & AV_CODEC_CAP_HYBRID)
        QTextStream(&str) << "hybrid ";
    if (!c->capabilities)
        QTextStream(&str) << "none";
    QTextStream(&str) << "\n";

    if (c->type == AVMEDIA_TYPE_VIDEO ||
        c->type == AVMEDIA_TYPE_AUDIO) {
        QTextStream(&str) << "    Threading capabilities: ";
        switch (c->capabilities & (AV_CODEC_CAP_FRAME_THREADS |
                                   AV_CODEC_CAP_SLICE_THREADS |
                                   AV_CODEC_CAP_AUTO_THREADS)) {
        case AV_CODEC_CAP_FRAME_THREADS |
             AV_CODEC_CAP_SLICE_THREADS: QTextStream(&str) << "frame and slice"; break;
        case AV_CODEC_CAP_FRAME_THREADS: QTextStream(&str) << "frame";           break;
        case AV_CODEC_CAP_SLICE_THREADS: QTextStream(&str) << "slice";           break;
        case AV_CODEC_CAP_AUTO_THREADS : QTextStream(&str) << "auto";            break;
        default:                         QTextStream(&str) << "none";            break;
        }
        QTextStream(&str) << "\n";
    }

    if (avcodec_get_hw_config(c, 0)) {
        QTextStream(&str) << "    Supported hardware devices: ";
        for (int i = 0;; i++) {
            const AVCodecHWConfig *config = avcodec_get_hw_config(c, i);
            if (!config)
                break;
            QTextStream(&str) << " " << av_hwdevice_get_type_name(config->device_type);
        }
        QTextStream(&str) << "\n";
    }

    if (c->supported_framerates) {
        const AVRational *fps = c->supported_framerates;

        QTextStream(&str) << "    Supported framerates:";
        while (fps->num) {
            sprintf(buffer, " %d/%d", fps->num, fps->den);
            QTextStream(&str) << buffer;
            fps++;
        }
        QTextStream(&str) << "\n";
    }
    PRINT_CODEC_SUPPORTED(c, pix_fmts, enum AVPixelFormat, "pixel formats",
                          AV_PIX_FMT_NONE, GET_PIX_FMT_NAME);
    PRINT_CODEC_SUPPORTED(c, supported_samplerates, int, "sample rates", 0,
                          GET_SAMPLE_RATE_NAME);
    PRINT_CODEC_SUPPORTED(c, sample_fmts, enum AVSampleFormat, "sample formats",
                          AV_SAMPLE_FMT_NONE, GET_SAMPLE_FMT_NAME);
    PRINT_CODEC_SUPPORTED(c, channel_layouts, uint64_t, "channel layouts",
                          0, GET_CH_LAYOUT_DESC);

    if (c->priv_class) {
        show_help_children(c->priv_class,
                           AV_OPT_FLAG_ENCODING_PARAM |
                           AV_OPT_FLAG_DECODING_PARAM);
    }

    return str;
}

void OptionPanel::show_help_codec(const char *name, int encoder)
{
    const AVCodecDescriptor *desc;
    const AVCodec *codec;

    if (!name) {
        MW->msg("No codec name specified.\n");
        return;
    }

    codec = encoder ? avcodec_find_encoder_by_name(name) :
                      avcodec_find_decoder_by_name(name);

    if (codec)
        cout << print_codec(codec).toStdString() << endl; //print_codec(codec);
    else if ((desc = avcodec_descriptor_get_by_name(name))) {
        int printed = 0;

        while ((codec = next_codec_for_id(desc->id, codec, encoder))) {
            printed = 1;
            //print_codec(codec);
            cout << print_codec(codec).toStdString() << endl;
        }

        if (!printed) {
            char buf[1024];
            sprintf(buf, "Codec '%s' is known to FFmpeg, "
                   "but no %s for it are available. FFmpeg might need to be "
                   "recompiled with additional external libraries.\n",
                   name, encoder ? "encoders" : "decoders");
            MW->msg(buf);
        }
    } else {
        char buf[256];
        sprintf(buf, "Codec '%s' is not recognized by FFmpeg.\n", name);
        MW->msg(buf);
    }

}

const QString OptionPanel::show_sample_fmts()
{
    int i;
    char fmt_str[128];
    QString str;

    for (i = -1; i < AV_SAMPLE_FMT_NB; i++) {
        char buffer[255];
        sprintf(buffer, "%s\n", av_get_sample_fmt_string(fmt_str, sizeof(fmt_str), (enum AVSampleFormat)i));
        QTextStream(&str) << buffer;
    }
    return str;
}

const QString OptionPanel::show_layouts()
{
    int i = 0;
    uint64_t layout, j;
    const char *name, *descr;

    char buffer[255];
    QString str;
    QTextStream(&str) << "Individual channels:\n"
                         "NAME           DESCRIPTION\n";
    for (i = 0; i < 63; i++) {
        name = av_get_channel_name((uint64_t)1 << i);
        if (!name)
            continue;
        descr = av_get_channel_description((uint64_t)1 << i);
        sprintf(buffer, "%-14s %s\n", name, descr);
        QTextStream(&str) << buffer;
    }
    QTextStream(&str) << "\nStandard channel layouts:\n"
                         "NAME           DECOMPOSITION\n";
    for (i = 0; !av_get_standard_channel_layout(i, &layout, &name); i++) {
        if (name) {
            sprintf(buffer, "%-14s ", name);
            QTextStream(&str) << buffer;
            for (j = 1; j; j <<= 1) {
                if ((layout & j)) {
                    sprintf(buffer, "%s%s", (layout & (j - 1)) ? "+" : "", av_get_channel_name(j));
                    QTextStream(&str) << buffer;
                }
            }
            QTextStream(&str) << "\n";
        }
    }
    return str;
}

const QString OptionPanel::show_pix_fmts()
{
    const AVPixFmtDescriptor *pix_desc = NULL;

    QString str;
    QTextStream(&str) << "Pixel formats:\n"
                         "I.... = Supported Input  format for conversion\n"
                         ".O... = Supported Output format for conversion\n"
                         "..H.. = Hardware accelerated format\n"
                         "...P. = Paletted format\n"
                         "....B = Bitstream format\n"
                         "FLAGS NAME            NB_COMPONENTS BITS_PER_PIXEL\n"
                         "-----\n";

    while ((pix_desc = av_pix_fmt_desc_next(pix_desc))) {
        enum AVPixelFormat av_unused pix_fmt = av_pix_fmt_desc_get_id(pix_desc);
        char buffer[255];
        sprintf(buffer, "%c%c%c%c%c %-16s       %d            %2d\n",
               sws_isSupportedInput (pix_fmt)              ? 'I' : '.',
               sws_isSupportedOutput(pix_fmt)              ? 'O' : '.',
               pix_desc->flags & AV_PIX_FMT_FLAG_HWACCEL   ? 'H' : '.',
               pix_desc->flags & AV_PIX_FMT_FLAG_PAL       ? 'P' : '.',
               pix_desc->flags & AV_PIX_FMT_FLAG_BITSTREAM ? 'B' : '.',
               pix_desc->name,
               pix_desc->nb_components,
               av_get_bits_per_pixel(pix_desc));
        QTextStream(&str) << buffer;
    }
    return str;
}

const QString OptionPanel::show_colors()
{
    const char *name;
    const uint8_t *rgb;
    int i;

    QString str;
    char buffer[256];
    sprintf(buffer, "%-32s #RRGGBB\n", "name");
    QTextStream(&str) << buffer;

    for (i = 0; name = av_get_known_color_name(i, &rgb); i++) {
        sprintf(buffer, "%-32s #%02x%02x%02x\n", name, rgb[0], rgb[1], rgb[2]);
        QTextStream(&str) << buffer;
    }

    return str;
}

const QString OptionPanel::show_bsfs()
{
    const AVBitStreamFilter *bsf = NULL;
    void *opaque = NULL;

    QString str;
    QTextStream(&str) << "Bitstream filters:\n";
    while ((bsf = av_bsf_iterate(&opaque)))
        QTextStream(&str) << bsf->name << "\n";
    QTextStream(&str) << "\n";

    return str;;
}

const QString OptionPanel::show_protocols()
{
    void *opaque = NULL;
    const char *name;

    QString str;

    QTextStream(&str) << "Supported file protocols:\nInput:\n";
    while ((name = avio_enum_protocols(&opaque, 0)))
        QTextStream(&str) << "  " <<  name << "\n";
    QTextStream(&str) << "Output:\n";
    while ((name = avio_enum_protocols(&opaque, 1)))
        QTextStream(&str) << " " << name << "\n";

    return str;
}

const QString OptionPanel::show_filters()
{
    const AVFilter *filter = NULL;
    char descr[64], *descr_cur;
    void *opaque = NULL;
    int i, j;
    const AVFilterPad *pad;

    QString str;
    QTextStream(&str) << "Filters:\n"
                         "  T.. = Timeline support\n"
                         "  .S. = Slice threading\n"
                         "  ..C = Command support\n"
                         "  A = Audio input/output\n"
                         "  V = Video input/output\n"
                         "  N = Dynamic number and/or type of input/output\n"
                         "  | = Source or sink filter\n";

    while ((filter = av_filter_iterate(&opaque))) {
        descr_cur = descr;
        for (i = 0; i < 2; i++) {
            if (i) {
                *(descr_cur++) = '-';
                *(descr_cur++) = '>';
            }
            pad = i ? filter->outputs : filter->inputs;
            for (j = 0; pad && avfilter_pad_get_name(pad, j); j++) {
                if (descr_cur >= descr + sizeof(descr) - 4)
                    break;
                *(descr_cur++) = get_media_type_char(avfilter_pad_get_type(pad, j));
            }
            if (!j)
                *(descr_cur++) = ((!i && (filter->flags & AVFILTER_FLAG_DYNAMIC_INPUTS)) ||
                                  ( i && (filter->flags & AVFILTER_FLAG_DYNAMIC_OUTPUTS))) ? 'N' : '|';
        }
        *descr_cur = 0;
        char buffer[256];
        sprintf(buffer, " %c%c%c %-17s %-10s %s\n",
               filter->flags & AVFILTER_FLAG_SUPPORT_TIMELINE ? 'T' : '.',
               filter->flags & AVFILTER_FLAG_SLICE_THREADS    ? 'S' : '.',
               filter->process_command                        ? 'C' : '.',
               filter->name, descr, filter->description);
        QTextStream(&str) << buffer;
    }
    return str;
}

const AVCodec * OptionPanel::next_codec_for_id(enum AVCodecID id, const AVCodec *prev, int encoder)
{
    while ((prev = av_codec_next(prev))) {
        if (prev->id == id &&
            (encoder ? av_codec_is_encoder(prev) : av_codec_is_decoder(prev)))
            return prev;
    }
    return NULL;
}

char OptionPanel::get_media_type_char(enum AVMediaType type)
{
    switch (type) {
        case AVMEDIA_TYPE_VIDEO:      return 'V';
        case AVMEDIA_TYPE_AUDIO:      return 'A';
        case AVMEDIA_TYPE_DATA:       return 'D';
        case AVMEDIA_TYPE_SUBTITLE:   return 'S';
        case AVMEDIA_TYPE_ATTACHMENT: return 'T';
        default:                      return '?';
    }
}

int OptionPanel::compare_codec_desc(const void *a, const void*b)
{
    const AVCodecDescriptor * const *da = (const AVCodecDescriptor * const *)a;
    const AVCodecDescriptor * const *db = (const AVCodecDescriptor * const *)b;

    return (*da)->type != (*db)->type ? FFDIFFSIGN((*da)->type, (*db)->type) :
           strcmp((*da)->name, (*db)->name);
}

unsigned OptionPanel::get_codecs_sorted(const AVCodecDescriptor ***rcodecs)
{
    const AVCodecDescriptor *desc = NULL;
    const AVCodecDescriptor **codecs;
    unsigned nb_codecs = 0, i = 0;

    while ((desc = avcodec_descriptor_next(desc)))
        nb_codecs++;
    if (!(codecs = (const AVCodecDescriptor**)av_calloc(nb_codecs, sizeof(*codecs)))) {
        cout << "Out of memory\n" << endl;
        exit_program(1);
    }
    desc = NULL;
    while ((desc = avcodec_descriptor_next(desc)))
        codecs[i++] = desc;
    av_assert0(i == nb_codecs);
    qsort(codecs, nb_codecs, sizeof(*codecs), compare_codec_desc);
    *rcodecs = codecs;
    return nb_codecs;
}

const QString OptionPanel::print_codecs(int encoder)
{
    const AVCodecDescriptor **codecs;
    unsigned nb_codecs = get_codecs_sorted(&codecs);

    QString str;
    QTextStream(&str) << (encoder ? "Encoders" : "Decoders") << "\n"
                      <<  " V..... = Video\n"
                          " A..... = Audio\n"
                          " S..... = Subtitle\n"
                          " .F.... = Frame-level multithreading\n"
                          " ..S... = Slice-level multithreading\n"
                          " ...X.. = Codec is experimental\n"
                          " ....B. = Supports draw_horiz_band\n"
                          " .....D = Supports direct rendering method 1\n"
                          " ------\n";


    for (unsigned i = 0; i < nb_codecs; i++) {
        const AVCodecDescriptor *desc = codecs[i];
        const AVCodec *codec = NULL;

        while ((codec = next_codec_for_id(desc->id, codec, encoder))) {
            QTextStream(&str) << " " << get_media_type_char(desc->type)
                              << ((codec->capabilities & AV_CODEC_CAP_FRAME_THREADS) ? "F" : ".")
                              << ((codec->capabilities & AV_CODEC_CAP_SLICE_THREADS) ? "S" : ".")
                              << ((codec->capabilities & AV_CODEC_CAP_EXPERIMENTAL)  ? "X" : ".")
                              << ((codec->capabilities & AV_CODEC_CAP_DRAW_HORIZ_BAND)?"B" : ".")
                              << ((codec->capabilities & AV_CODEC_CAP_DR1)           ? "D" : ".");

            char buffer[256];
            sprintf(buffer, " %-20s %s", codec->name, codec->long_name ? codec->long_name : "");
            QTextStream(&str) << buffer;
            if (strcmp(codec->name, desc->name))
                QTextStream(&str) << "(codec " << desc->name << ")";
            QTextStream(&str) << "\n";
        }

    }

    av_free(codecs);
    return str;
}

const QString OptionPanel::show_help_options(const OptionDef *options, const char *msg, int req_flags,
                       int rej_flags, int alt_flags)
{
    const OptionDef *po;
    int first;

    char buffer[256];
    QString str;

    first = 1;
    for (po = options; po->name; po++) {
        char buf[64];

        if (((po->flags & req_flags) != req_flags) ||
            (alt_flags && !(po->flags & alt_flags)) ||
            (po->flags & rej_flags))
            continue;

        if (first) {
            QTextStream(&str) << msg << "\n";
            first = 0;
        }
        av_strlcpy(buf, po->name, sizeof(buf));
        if (po->argname) {
            av_strlcat(buf, " ", sizeof(buf));
            av_strlcat(buf, po->argname, sizeof(buf));
        }
        sprintf(buffer, "-%-17s  %s\n", buf, po->help);
        QTextStream(&str) << buffer;
    }
    QTextStream(&str) << "\n";
    return str;
}

void OptionPanel::showParameterDialog()
{
    parameterDialog->show();
}

