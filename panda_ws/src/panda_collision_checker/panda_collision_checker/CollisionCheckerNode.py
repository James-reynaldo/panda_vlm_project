import rclpy
from rclpy.node import Node

from panda_interfaces.srv import CollisionCheck

from .Panda_scene import PandaScene
from .collision_checker import CollisionChecker


class CollisionCheckerNode(Node):

    def __init__(self):
        super().__init__("collision_checker")

        self.env = PandaScene(
            robots="Panda",
            render=True,
        )

        self.env.reset()

        self.checker = CollisionChecker(self.env)

        self.service = self.create_service(
            CollisionCheck,
            "/collision_check",
            self.check_collision,
        )

        self.get_logger().info(
            "Collision checker ready."
        )

    def check_collision(self, request, response):

        collision = self.checker.check_configuration(
            request.joint_positions
        )

        response.collision_free = not collision

        return response


def main():
    rclpy.init()

    node = CollisionCheckerNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()