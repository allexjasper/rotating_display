#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <boost/asio.hpp>
#include <boost/regex.hpp>
#include <chrono>
#include <mutex>
#include <fstream>
#include <optional>
#include <deque>
#include <boost/lexical_cast.hpp>
#include <tuple>
#include <boost/program_options.hpp>

#include <SDL.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <windows.h>
#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sources/logger.hpp>
#include <boost/log/sources/severity_logger.hpp>
#include <boost/log/sources/record_ostream.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/trivial.hpp>



using namespace boost::asio;
using namespace std;
namespace po = boost::program_options;
namespace logging = boost::log;
namespace src = boost::log::sources;
namespace sinks = boost::log::sinks;
namespace keywords = boost::log::keywords;


const size_t inboundBufferSize = 20000; // Modify as needed
const size_t rotationsBufferSize  = 50; // Modify as needed
const size_t plotBufferSize = 30000; // Modify as needed

typedef std::chrono::time_point<std::chrono::steady_clock> Timestamp;
typedef std::tuple<float, Timestamp, Timestamp> Sample; // angle, sensor time, local time
typedef  std::vector<Sample>  RawData;
typedef  std::deque<Sample>  RawDataDeueue;

src::severity_logger< logging::trivial::severity_level > lg;

// Initialize Boost::Log
void initLogging() {
    logging::add_file_log
    (
        keywords::file_name = "rotating_display_%N.log",
        keywords::rotation_size = 10 * 1024 * 1024,
        keywords::time_based_rotation = sinks::file::rotation_at_time_point(0, 0, 0),
        keywords::format = "[%TimeStamp%]: %Message%"
    );
}




class AngleTimeSeries {
public:
    std::chrono::milliseconds _duration;

    // Rotation constructor default values
    AngleTimeSeries() : _duration(0)
    {
	}
    

    AngleTimeSeries(RawData& samples) {
        this->samples = samples;
    }

    RawData samples;

    // Compute start_time and duration dynamically from first and last sample
    Timestamp start_time() const {
        return std::get<1>(samples.front());
    }

    std::chrono::milliseconds duration() const {
        return _duration;//std::chrono::duration_cast<std::chrono::milliseconds>(std::get<1>(samples.back()) - std::get<1>(samples.front()));
    }

    // compute mediant of vecotr of floats
    static float median(std::vector<float> v)
    {
		size_t n = v.size() / 2;
		std::nth_element(v.begin(), v.begin() + n, v.end());
		return v[n];
	}

    static AngleTimeSeries computeAverageTimeSeries(const std::vector<AngleTimeSeries>& input)
    {
        AngleTimeSeries averageTimeSeries;
        
		// Compute the average time series
        auto s = input.front().samples.size();
		averageTimeSeries.samples.resize(s);
        for (size_t i = 0; i < averageTimeSeries.samples.size(); i++)
        {
            std::vector<float> angles;
			
            for (const AngleTimeSeries& timeSeries : input)
            {
                if (timeSeries.samples.size() > i)
                {
					angles.push_back(std::get<0>(timeSeries.samples[i]));
				}
			}

			averageTimeSeries.samples[i] = Sample(median(angles), std::get<1>(input.front().samples[i]), std::get<2>(input.front().samples[i]));
		
		}
        averageTimeSeries._duration = input.front().duration();
		return averageTimeSeries;
    }




    // Resample the rotation data to evenly space 1 ms buckets with interpolation
    void resample() {
        if (samples.empty()) {
            return; // Nothing to resample
        }

        std::chrono::milliseconds bucket_duration(1); // 1 ms bucket duration

        Timestamp start_time = std::get<1>(samples.front());
        Timestamp end_time = std::get<1>(samples.back());

        std::vector<Sample> resampled_data;
        Timestamp current_time = start_time;

        for (size_t i = 0; i < samples.size() - 1; ++i) {
            Timestamp timestamp1 = std::get<1>(samples[i]);
            Timestamp timestamp2 = std::get<1>(samples[i + 1]);

            // Interpolate between the two adjacent data points
            while (current_time <= timestamp2) {
                float t = static_cast<float>((current_time - timestamp1).count()) / (timestamp2 - timestamp1).count();
                float interpolated_angle = std::get<0>(samples[i]) + t * (std::get<0>(samples[i + 1]) - std::get<0>(samples[i]));
                Timestamp interpolated_local_time = current_time;
                resampled_data.push_back(std::make_tuple(interpolated_angle, current_time, interpolated_local_time));
                current_time += bucket_duration;
            }
        }

        samples = resampled_data; // Replace the original data with the resampled data
    }


    // Method to make sensor timestamps relative to the start timestamp of the rotation
    void makeSensorTimestampsRelative() {
        if (samples.empty()) {
            return; // Nothing to adjust
        }

        Timestamp startTimestamp = std::get<1>(samples.front());

        for (Sample& sample : samples) {
            Timestamp sensorTimestamp = std::get<1>(sample);
            sample = Sample(std::get<0>(sample), sensorTimestamp - startTimestamp, std::get<2>(sample));
        }
    }

    // extend the rotation by 20% at the end by adding the samples from the beginning at end while adjusting the timestamps
    void extendRotation() {
        if (samples.empty()) {
            return; // Nothing to extend
        }

        size_t extensionSize = 40;//static_cast<size_t>(samples.size() * 0.5); // Extend by 20%


        // Extract the first 'extensionSize' samples for extension
        RawData extensionSamples(samples.begin(), samples.begin() + extensionSize);

        // Calculate the time duration of the extension
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::get<1>(samples.back()) - std::get<1>(samples.front())) +
            std::chrono::duration_cast<std::chrono::milliseconds>(std::get<1>(samples[1]) - std::get<1>(samples[0]));


        // Adjust timestamps of the extension samples
        for (Sample& sample : extensionSamples) {
            //std::get<1>(sample) += duration - std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::milliseconds(10));
            //std::get<2>(sample) += duration - std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::milliseconds(10));
            std::get<1>(sample) += duration - std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::milliseconds(10));
            std::get<2>(sample) += duration - std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::milliseconds(10));
        }

        // Add the extension samples to the end of the rotation
        samples.insert(samples.end(), extensionSamples.begin(), extensionSamples.end());
    }


    void generateGnuplotDataFile(const std::string& filename) {
        std::ofstream outputFile(filename);
        if (!outputFile.is_open()) {
            std::cerr << "Failed to open the file for writing: " << filename << std::endl;
            return;
        }

        for (const Sample& sample : samples) {
            // Format the data as "timestamp angle"
            outputFile << std::get<1>(sample).time_since_epoch().count() << " " << std::get<0>(sample) << std::endl;
        }

        outputFile.close();
    }

    void generateGnuplotDataFileWithMatch(const std::string& filename, const AngleTimeSeries& shorter_sub_time_series, int offset) {
        std::ofstream outputFile(filename);
        if (!outputFile.is_open()) {
            std::cerr << "Failed to open the file for writing: " << filename << std::endl;
            return;
        }

        for (int i = 0; i < samples.size(); i++) {
            // Format the data as "timestamp angle"
            auto sample = samples[i];
            if(i >= offset && i < offset + shorter_sub_time_series.samples.size())
                outputFile << std::get<1>(sample).time_since_epoch().count() << " " << std::get<0>(sample) << " " << std::get<0>(shorter_sub_time_series.samples[i - offset]) << std::endl;
			else
                outputFile << std::get<1>(sample).time_since_epoch().count() << " " << std::get<0>(sample) << std::endl;
        }

        outputFile.close();
    }


    std::pair<float, int> findBestMatch(const AngleTimeSeries& shorter_sub_time_series) const {
        if (shorter_sub_time_series.samples.empty() || samples.empty() || shorter_sub_time_series.samples.size() > samples.size()) {
            // Invalid input
            return std::make_pair(std::numeric_limits<float>::max(), -1);
        }

        float best_norm = std::numeric_limits<float>::max();
        int best_match_index = -1;

        for (int i = 0; i <= static_cast<int>(samples.size()) - static_cast<int>(shorter_sub_time_series.samples.size()); ++i) {
            float norm = calculateNormDifference(shorter_sub_time_series, i);
            if (norm < best_norm) {
                best_norm = norm;
                best_match_index = i;
            }
        }

        return std::make_pair(best_norm, best_match_index);
    }


    // Store the last N samples and resample
    void storeAndResampleLastNSamples(const RawData& inboundData, int numSamplesToStore) {
        if (inboundData.size() < numSamplesToStore) {
            BOOST_LOG_TRIVIAL(error) << "Not enough samples to estimate position.";
            return;
        }

        // Extract the last N samples
        samples.assign(inboundData.end() - numSamplesToStore, inboundData.end());

        // Resample the stored samples
        resample();
    }

    // Estimate position using the reference rotation
    std::pair<float, int> estimate_position(const AngleTimeSeries& referenceRotation) const {
        

        return referenceRotation.findBestMatch(*this);
    }

private:
    float calculateNormDifference(const AngleTimeSeries& shorter_sub_time_series, int i) const {
        if (i < 0 || i + shorter_sub_time_series.samples.size() > samples.size()) {
            return std::numeric_limits<float>::max();
        }

        // Calculate the L2 (Euclidean) norm between the aligned sub-sequences
        float norm = 0.0f;
        for (size_t j = 0; j < shorter_sub_time_series.samples.size(); ++j) {
            float diff = std::get<0>(samples[i + j]) - std::get<0>(shorter_sub_time_series.samples[j]);
            norm += diff * diff;
        }
        norm = std::sqrt(norm);

        return norm;
    }


};


RawDataDeueue inbound;
std::mutex lock_inbound;

std::optional<AngleTimeSeries> referenceRotation;
std::mutex lock_referenceRotation;


class RefRotationToLocal
{
public:
    AngleTimeSeries reference_rotation;
    Timestamp ref_timestamp; // timestamp in the reference rotation (after normalization)
    Timestamp local_timestamp; // timestamp of local time matching the reference rotation timestamp
};

std::mutex lock_ref_rotation_to_local;
std::optional <RefRotationToLocal> ref_rotation_to_local;

float magnetOffset = 0.0;
std::chrono::milliseconds clock_offset = std::chrono::milliseconds(0);

void receive_sensor_data_udp() {
    io_service io_service;
    ip::udp::endpoint endpoint(ip::udp::v4(), 20001);
    ip::udp::socket socket(io_service, endpoint);

    while (true) {
        char data[256];

        // fill data with zeros
        for (int i = 0; i < 256; i++) {
			        data[i] = 0;
		        }   

        ip::udp::endpoint sender_endpoint;
        socket.receive_from(buffer(data, 256), sender_endpoint);

        std::string decodedPacket(data);
        boost::regex pattern("(\\d*), (\\d*)");
        boost::smatch match;

        if (boost::regex_match(decodedPacket, match, pattern)) {
            float angle = static_cast<float>(std::stoi(match[1])) / 1000.0f - magnetOffset;
            // extract from match[2] the timestamp using lexical_cast
            long long timestamp = std::stoll(match[2]);
            Timestamp ts = std::chrono::time_point<std::chrono::steady_clock>(std::chrono::milliseconds(timestamp));

            Sample tuple = { angle, ts, std::chrono::steady_clock::now() };
           

            std::lock_guard<std::mutex> guard(lock_inbound);
            inbound.push_back(tuple);
            if (inbound.size() > inboundBufferSize)
                inbound.pop_front();
            
        }
    }
}


vector<AngleTimeSeries> segment_data(const RawData& data) {
    Sample prevSample;
    AngleTimeSeries curRotation;
    vector<AngleTimeSeries> rotations;
    float minAngle = std::numeric_limits<float>::max();
    float maxAngle = std::numeric_limits<float>::min();

    for (const auto& sample : data) {
        float angle = std::get<0>(sample);

        // Update min and max angles
        if (angle < minAngle) {
            minAngle = angle;
        }
        if (angle > maxAngle) {
            maxAngle = angle;
        }

        if (!curRotation.samples.empty()) {
            // Start of rotation is when the angle increases and crosses 0, save the current rotation and start a new one
            if (std::get<0>(prevSample) < angle && std::get<0>(prevSample) <= 0.0 && angle > 0.0) {
                // Is it a full rotation?
                if (std::abs(std::get<0>(curRotation.samples.front())) < 0.5 && std::abs(std::get<0>(curRotation.samples.back())) < 0.5) {
                    // Start and end angle are close to 0
                    curRotation._duration = std::chrono::milliseconds(std::chrono::duration_cast<std::chrono::milliseconds>(std::get<1>(sample) - std::get<1>(curRotation.samples.front())).count());
                    rotations.push_back(curRotation);
                }
                curRotation = AngleTimeSeries();
            }
        }

        curRotation.samples.push_back(sample);
        prevSample = sample;
    }

    // Print the min and max angles
    BOOST_LOG_TRIVIAL(info) << "min angle observed while segmenting: " << minAngle << std::endl;
    BOOST_LOG_TRIVIAL(info) << "max angle: observed while segmenting: " << maxAngle << std::endl;

    return rotations;
}


AngleTimeSeries findNMedianDuration(const vector<AngleTimeSeries>& rotations, int n) {
    if (rotations.empty()) {
        throw runtime_error("No rotations to find the median duration from.");
    }

    vector<AngleTimeSeries> rotations_copy = rotations;


    // Sort the rotations by duration
    sort(rotations_copy.begin(), rotations_copy.end(), [](const AngleTimeSeries& a, const AngleTimeSeries& b) {
		return a.duration() < b.duration();
	});

    std::vector<AngleTimeSeries> rotations_median(rotations_copy.begin() + rotations_copy.size() / 2 - n / 2, rotations_copy.begin() + rotations_copy.size() / 2 + n / 2);
    return AngleTimeSeries::computeAverageTimeSeries(rotations_median);

    //return rotations_copy[rotations_copy.size() / 2]; // Return the rotation with the median duration
}


void segment_data_periodically() {
    while (true) {
        // Call segment_data function every 10 seconds
        RawData rd;
        
        {
            std::lock_guard<std::mutex> guard(lock_inbound);
            rd.assign(inbound.begin(), inbound.end());
        }
        std::vector<AngleTimeSeries> rotations = segment_data(rd);
        if (rotations.size() < 10) {
            std::this_thread::sleep_for(std::chrono::seconds(10));
            continue;
		}

        
        AngleTimeSeries reference_rotation_local = findNMedianDuration(rotations, 5);
        reference_rotation_local.makeSensorTimestampsRelative();
        reference_rotation_local.extendRotation();
        reference_rotation_local.resample();
        reference_rotation_local.generateGnuplotDataFile("reference_rotation.dat");
        
        {
            BOOST_LOG_TRIVIAL(info) << "Reference rotation duration: " << reference_rotation_local.duration().count() << " ms" << std::endl;
			std::lock_guard<std::mutex> guard(lock_referenceRotation);
			//if(!referenceRotation.has_value()) // todo: activate later update of reference rotation
                referenceRotation = reference_rotation_local;
		}

        std::this_thread::sleep_for(std::chrono::seconds(10));
    }
}

int numMatches = 0;

// Function to perform position estimation every 5 seconds
void performPositionEstimationPeriodically() {
    int numSamplesToStore = 20; // about 500 ms of data
    AngleTimeSeries angleTimeSeries;
    while (true) {
        RawData inboundDataCopy;
        {
            std::lock_guard<std::mutex> guard(lock_inbound);
            inboundDataCopy.assign(inbound.begin(), inbound.end());
        }

        std::optional<AngleTimeSeries> referenceRotationCopy;
        {
			std::lock_guard<std::mutex> guard(lock_referenceRotation);
            referenceRotationCopy = referenceRotation;
		}

        

        // Estimate position using the reference rotation
        if (!referenceRotationCopy.has_value())
        {
            BOOST_LOG_TRIVIAL(trace) << "Reference rotation is not available yet.";
		}
        else
        {
            // Store and resample the last N samples
            angleTimeSeries.storeAndResampleLastNSamples(inboundDataCopy, numSamplesToStore);

            std::pair<float, int> result = angleTimeSeries.estimate_position(*referenceRotationCopy);

            // Display the result (you can replace this with your specific processing)
            BOOST_LOG_TRIVIAL(trace) << "Estimated Position: Norm = " << result.first << ", Index = " << result.second
                                                                      << " num match =" << numMatches << ::endl;

            // Generate gnuplot data file for the reference rotation and the inbound data
            std::string numStr = boost::lexical_cast<std::string>(numMatches);
            //referenceRotationCopy->generateGnuplotDataFileWithMatch("inbound_data_with_match_" + numStr, angleTimeSeries, result.second);
            numMatches++;

            RefRotationToLocal ref_rotation_to_local_local;
            ref_rotation_to_local_local.reference_rotation = *referenceRotationCopy;
            ref_rotation_to_local_local.ref_timestamp = std::get<1>(referenceRotationCopy->samples[result.second + angleTimeSeries.samples.size()]);
            ref_rotation_to_local_local.local_timestamp = std::get<2>(inboundDataCopy.back());

            std::lock_guard<std::mutex> guard(lock_ref_rotation_to_local);
            ref_rotation_to_local = ref_rotation_to_local_local;
        }

        // Sleep for 5 seconds before the next estimation
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

typedef std::tuple<Timestamp, std::optional<float>, std::optional<float>> RenderPoint; // timestamp, angle raw, angle corrected
std::deque<RenderPoint> plot_data; // timestamp, angle raw, angle corrected
std::mutex lock_plot_data;

void write_plot_data_to_file() {
    SetThreadPriority(GetCurrentThread(), THREAD_BASE_PRIORITY_MIN);
    for (int i = 0; true; i++)
    {
        std::this_thread::sleep_for(std::chrono::seconds(10));

        std::ofstream outputFile("plot_" + boost::lexical_cast<std::string>(i));
        if (!outputFile.is_open()) {
            std::cerr << "Failed to open the file for writing: "  << std::endl;
            return;
        }

        std::deque<RenderPoint> plot_data_copy;
        {
			std::lock_guard<std::mutex> guard(lock_plot_data);
			plot_data_copy = plot_data;
		}

        for (const RenderPoint& plot_point : plot_data_copy) {
            // Format the data as "timestamp angle_raw angle_corrected"
            Timestamp timestamp = std::get<0>(plot_point);
            if(!std::get<1>(plot_point) || !std::get<2>(plot_point))
                continue;
            float angle_raw = *std::get<1>(plot_point);
            float angle_corrected = *std::get<2>(plot_point);
            outputFile << timestamp.time_since_epoch().count() << " " << angle_raw << " " << angle_corrected << std::endl;
        }

        outputFile.close();
    }
}

std::chrono::milliseconds transmission_delay;

RenderPoint calc_render_point()
{
    RenderPoint plot_point;

    Timestamp curTime = std::chrono::steady_clock::now();
    std::get<0>(plot_point) = curTime;

    Sample latest_sample;
    {
        std::lock_guard<std::mutex> guard(lock_inbound);
        if (inbound.size() > 0)
        {
            latest_sample = inbound.back();

        }
        else
        {
            return plot_point;
        }
    }

    // angle raw
    std::get<1>(plot_point) = std::get<0>(latest_sample);

    std::lock_guard<std::mutex> guard2(lock_ref_rotation_to_local);
    if (!ref_rotation_to_local)
    {
        return plot_point;
    }

    // how much time has locally elapsed since the last match between reference rotation and local time was computed
    std::chrono::milliseconds time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(curTime - ref_rotation_to_local->local_timestamp);

    // compute match for current time in reference rotation
    Timestamp curTSInRef = (ref_rotation_to_local->ref_timestamp + time_elapsed + transmission_delay);
    auto duration = (curTSInRef - std::get<1>(ref_rotation_to_local->reference_rotation.samples[0]));
    // convert duration from chrono duration to the count of milliscends
    long long duration_count_ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
    auto index = duration_count_ms % ref_rotation_to_local->reference_rotation.duration().count();

    // angle corrected
    std::get<2>(plot_point) = std::get<0>(ref_rotation_to_local->reference_rotation.samples[index]);
    return plot_point;
}

// function that stores angles as they would be rendered in an array for plotting
void render_to_plot()
{
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
    while (true)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        RenderPoint plot_point = calc_render_point();

        {
		        
            std::lock_guard<std::mutex> guard3(lock_plot_data);
            plot_data.push_back(plot_point);
            if(plotBufferSize < plot_data.size())
				plot_data.pop_front();
        }
    }
}


int render_opengl()
{
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
    
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        fprintf(stderr, "SDL could not initialize! SDL_Error: %s\n", SDL_GetError());
        return 1;
    }

    SDL_DisplayMode dm;
    if (SDL_GetCurrentDisplayMode(0, &dm) != 0) {
        SDL_Quit();
        return 1;
    }

    int screenWidth = dm.w * 0.7;
    int screenHeight = dm.h *0.7;

    //SDL_Window* window = SDL_CreateWindow("OpenGL Window", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, screenWidth, screenHeight, SDL_WINDOW_FULLSCREEN | SDL_WINDOW_OPENGL);
    SDL_Window* window = SDL_CreateWindow("OpenGL Window", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, screenWidth, screenHeight, SDL_WINDOW_OPENGL);
    if (window == nullptr) {
        fprintf(stderr, "Window could not be created! SDL_Error: %s\n", SDL_GetError());
        SDL_Quit();
        return 1;
    }

    SDL_GLContext context = SDL_GL_CreateContext(window);
    if (context == nullptr) {
        fprintf(stderr, "OpenGL context could not be created! SDL_Error: %s\n", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glViewport(0, 0, screenWidth, screenHeight);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, screenWidth, screenHeight, 0, -10, 10);

    float prevAngle = 0.0f;
    float angle = 0.0f;

    while (1) {
        SDL_Event e;
        if (SDL_PollEvent(&e) && e.type == SDL_QUIT) {
            break;
        }

        RenderPoint plot_point = calc_render_point();

        if (std::get<2>(plot_point))
        {
            angle = *std::get<2>(plot_point);
        }
        else if (std::get<1>(plot_point))
        {
			angle = *std::get<1>(plot_point);
		}
        else
        {
			angle = 0.0f;
		}

        // Render the scene
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glLoadIdentity();
        glOrtho(0, screenWidth, screenHeight, 0, -10, 10);
        glTranslatef(screenWidth / 2, screenHeight / 2, 0.0f);
        glRotatef(-angle, 0, 0, 1);

        glBegin(GL_QUADS);
        glColor3f(1, 0, 0);
        glVertex2f(-600, -400);
        glVertex2f(-600, 400);
        glVertex2f(-10, 400);
        glVertex2f(-10, -400);
        glEnd();

        glBegin(GL_QUADS);
        glColor3f(1, 0, 0);
        glVertex2f(10, -400);
        glVertex2f(10, 400);
        glVertex2f(600, 400);
        glVertex2f(600, -400);
        glEnd();

        SDL_GL_SwapWindow(window);
        SDL_Delay(10); // Equivalent to pygame.time.wait(10)

        // Update angle or other logic as needed
        if(std::abs(angle - prevAngle) > 0.5)
			BOOST_LOG_TRIVIAL(trace) << "Angle jump: " << angle - prevAngle << std::endl;

        prevAngle = angle;
        // Add your logic here to update 'angle'
    }

    SDL_GL_DeleteContext(context);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}




int main(int argc, char** argv) {

    // Initialize logging
    initLogging();
    logging::add_common_attributes();

    
    BOOST_LOG_TRIVIAL(trace) << "Client starting";
    

    float magnet_offset;
    int time_offset;  // Change to integer

    po::options_description desc("Allowed options");
    desc.add_options()
        ("magnet_offset,m", po::value<float>(&magnet_offset)->default_value(0.0), "Magnet Offset (float)")
        ("time_offset,t", po::value<int>(&time_offset)->default_value(0), "Time Offset (integer)");  // Change to integer

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("magnet_offset")) {
        BOOST_LOG_TRIVIAL(trace)  << "Magnet Offset: " << magnet_offset << std::endl;
    }

    if (vm.count("time_offset")) {
        BOOST_LOG_TRIVIAL(trace) << "Time Offset: " << time_offset << std::endl;
    }

    magnetOffset = magnet_offset;
    transmission_delay = std::chrono::milliseconds(time_offset);  // Change to integer


    // Start the receive_sensor_data function in a separate thread
    std::thread sensor_data_thread(receive_sensor_data_udp);

    // Start the segment_data_periodically function in a separate thread
    std::thread segment_data_thread(segment_data_periodically);

    // Create a thread to perform position estimation periodically
    std::thread estimationThread(performPositionEstimationPeriodically);

    // Create a thread to perform position estimation periodically
    std::thread render_plotThread(render_to_plot);

    std::thread write_plotThread(write_plot_data_to_file);

    std::thread render_openGLThread(render_opengl);



    // Wait for the threads to finish
    sensor_data_thread.join();
    segment_data_thread.join();
    //clock_offset_thread.join();
    estimationThread.join();
    render_plotThread.join();
    render_openGLThread.join();

    return 0;
}
