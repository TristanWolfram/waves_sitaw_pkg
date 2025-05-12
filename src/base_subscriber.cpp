#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

static const std::string POINTCLOUD_TOPIC = "/sim_LiDAR_depth/points";
static const std::string IMAGE_TOPIC = "/sim_cam_color_0/image_color";

using namespace std::placeholders;

class BaseSub : public rclcpp::Node
{
public:
  BaseSub() : Node("base_sub_node")
  {
    // Initialize message_filters subscribers for each topic.
    sub_pointcloud_.subscribe(this, POINTCLOUD_TOPIC);
    sub_image_.subscribe(this, IMAGE_TOPIC);

    // Create an approximate time synchronizer with a queue size of 10.
    sync_.reset(new Sync(MySyncPolicy(10), sub_pointcloud_, sub_image_));
    sync_->registerCallback(std::bind(&BaseSub::callback, this, _1, _2));

    RCLCPP_INFO(this->get_logger(), "Base Node started. Waiting for synchronized messages...");
  }

private:
  // This callback is triggered when a pair of pointcloud and image messages
  // have been received that are close in time.
  void callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr pointcloud_msg,
                const sensor_msgs::msg::Image::ConstSharedPtr image_msg)
  {

    uint32_t num_points = pointcloud_msg->width * pointcloud_msg->height;

    uint32_t image_width = image_msg->width;
    uint32_t image_height = image_msg->height;

    RCLCPP_INFO(this->get_logger(),
      "Received synchronized messages: PointCloud with %u points, Image size: %ux%u",
      num_points, image_width, image_height);
  }

  // Message filter subscribers for the pointcloud and image topics.
  message_filters::Subscriber<sensor_msgs::msg::PointCloud2> sub_pointcloud_;
  message_filters::Subscriber<sensor_msgs::msg::Image> sub_image_;

  // Define the sync policy using approximate time.
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::PointCloud2,
                                                          sensor_msgs::msg::Image> MySyncPolicy;
  typedef message_filters::Synchronizer<MySyncPolicy> Sync;
  std::shared_ptr<Sync> sync_;
};



int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<BaseSub>());
  rclcpp::shutdown();
  return 0;
}
