#include <memory>
#include <string>

#include <gz/sim/System.hh>
#include <gz/sim/Model.hh>
#include <gz/sim/Util.hh>
#include <gz/plugin/Register.hh>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>

namespace smart_city
{
class ModelWorldPosePublisher:
    public gz::sim::System,
    public gz::sim::ISystemConfigure,
    public gz::sim::ISystemPostUpdate
{
public:
    void Configure(
    const gz::sim::Entity &_entity,
    const std::shared_ptr<const sdf::Element> &,
    gz::sim::EntityComponentManager &_ecm,
    gz::sim::EventManager &) override
    {
        this->model = gz::sim::Model(_entity);
        this->modelName = this->model.Name(_ecm);

        if (!rclcpp::ok())
        {
            rclcpp::init(0, nullptr);
        }

        this->node = std::make_shared<rclcpp::Node>(
            this->modelName + "_world_pose_publisher"
        );

        std::string topic = "/gazebo/model_pose/" + this->modelName;

        this->pub = this->node->create_publisher<geometry_msgs::msg::PoseStamped>(
            topic,
            10
        );

        RCLCPP_INFO(
            this->node->get_logger(),
            "Publishing world pose for model '%s' on topic '%s'",
            this->modelName.c_str(),
            topic.c_str()
        );
    }

    void PostUpdate(
        const gz::sim::UpdateInfo &,
        const gz::sim::EntityComponentManager &_ecm) override
    {
        if (!this->node || !this->pub)
            return;

        auto pose = gz::sim::worldPose(this->model.Entity(), _ecm);

        geometry_msgs::msg::PoseStamped msg;
        msg.header.frame_id = "world";
        msg.header.stamp = this->node->now();

        msg.pose.position.x = pose.Pos().X();
        msg.pose.position.y = pose.Pos().Y();
        msg.pose.position.z = pose.Pos().Z();

        msg.pose.orientation.x = pose.Rot().X();
        msg.pose.orientation.y = pose.Rot().Y();
        msg.pose.orientation.z = pose.Rot().Z();
        msg.pose.orientation.w = pose.Rot().W();

        this->pub->publish(msg);

        rclcpp::spin_some(this->node);
    }

private:
    gz::sim::Model model;
    std::string modelName;

    rclcpp::Node::SharedPtr node;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub;
};
}

GZ_ADD_PLUGIN(
    smart_city::ModelWorldPosePublisher,
    gz::sim::System,
    smart_city::ModelWorldPosePublisher::ISystemConfigure,
    smart_city::ModelWorldPosePublisher::ISystemPostUpdate
)