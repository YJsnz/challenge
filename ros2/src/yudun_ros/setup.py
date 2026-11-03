from setuptools import setup
setup(name="yudun_ros", version="0.1.0", packages=["yudun_ros"],
      data_files=[("share/ament_index/resource_index/packages", ["resource/yudun_ros"]),
                  ("share/yudun_ros", ["package.xml"])],
      install_requires=["setuptools"], zip_safe=True,
      maintainer="Yudun team", maintainer_email="maintainer@example.invalid",
      description="Yudun sensor and fusion nodes", license="Proprietary",
      entry_points={"console_scripts": ["fusion = yudun_ros.fusion:main", "sensors = yudun_ros.sensors:main"]})
