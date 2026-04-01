import collections
import enum
import os
from typing import Any, Iterable, List, Mapping, NamedTuple, Optional, Union

import numpy as np
import tensorflow as tf

from google.protobuf.internal import containers
from google.protobuf import descriptor
from google.protobuf import message
# resources dependency
# pylint: disable=unused-import
from mediapipe.calculators.core import constant_side_packet_calculator_pb2
from mediapipe.calculators.image import image_transformation_calculator_pb2
from mediapipe.calculators.tensor import tensors_to_detections_calculator_pb2
from mediapipe.calculators.util import landmarks_smoothing_calculator_pb2
from mediapipe.calculators.util import logic_calculator_pb2
from mediapipe.calculators.util import thresholding_calculator_pb2
from mediapipe.framework import calculator_pb2
from mediapipe.framework.formats import body_rig_pb2
from mediapipe.framework.formats import classification_pb2
from mediapipe.framework.formats import detection_pb2
from mediapipe.framework.formats import landmark_pb2
from mediapipe.framework.formats import rect_pb2
from mediapipe.modules.objectron.calculators import annotation_data_pb2
from mediapipe.modules.objectron.calculators import lift_2d_frame_annotation_to_3d_calculator_pb2
# pylint: enable=unused-import
from mediapipe.python import packet_creator
from mediapipe.python import packet_getter
from mediapipe.python._framework_bindings import calculator_graph
from mediapipe.python._framework_bindings import image_frame
from mediapipe.python._framework_bindings import packet
from mediapipe.python._framework_bindings import resource_util
from mediapipe.python._framework_bindings import validated_graph_config

RGB_CHANNELS = 3
# TODO: Enable calculator options modification for more calculators.
CALCULATOR_TO_OPTIONS = {
    'ConstantSidePacketCalculator':
        constant_side_packet_calculator_pb2.ConstantSidePacketCalculatorOptions,
    'ImageTransformationCalculator':
        image_transformation_calculator_pb2
        .ImageTransformationCalculatorOptions,
    'LandmarksSmoothingCalculator':
        landmarks_smoothing_calculator_pb2.LandmarksSmoothingCalculatorOptions,
    'LogicCalculator':
        logic_calculator_pb2.LogicCalculatorOptions,
    'ThresholdingCalculator':
        thresholding_calculator_pb2.ThresholdingCalculatorOptions,
    'TensorsToDetectionsCalculator':
        tensors_to_detections_calculator_pb2
        .TensorsToDetectionsCalculatorOptions,
    'Lift2DFrameAnnotationTo3DCalculator':
        lift_2d_frame_annotation_to_3d_calculator_pb2
        .Lift2DFrameAnnotationTo3DCalculatorOptions,
}


def type_names_from_oneof(oneof_type_name: str) -> Optional[List[str]]:
    if oneof_type_name.startswith('OneOf<') and oneof_type_name.endswith('>'):
        comma_separated_types = oneof_type_name[len('OneOf<'):-len('>')]
        return [n.strip() for n in comma_separated_types.split(',')]
    return None


# TODO: Support more packet data types, such as "Any" type.
@enum.unique
class PacketDataType(enum.Enum):
    """The packet data types supported by the SolutionBase class."""
    STRING = 'string'
    BOOL = 'bool'
    BOOL_LIST = 'bool_list'
    INT = 'int'
    INT_LIST = 'int_list'
    FLOAT = 'float'
    FLOAT_LIST = 'float_list'
    AUDIO = 'matrix'
    IMAGE = 'image'
    IMAGE_LIST = 'image_list'
    IMAGE_FRAME = 'image_frame'
    PROTO = 'proto'
    PROTO_LIST = 'proto_list'
    TENSOR = 'tensor' #Added tensor type

    @staticmethod
    def from_registered_name(registered_name: str) -> 'PacketDataType':
        try:
            return NAME_TO_TYPE[registered_name]
        except KeyError as e:
            names = type_names_from_oneof(registered_name)
            if names:
                for n in names:
                    if n in NAME_TO_TYPE.keys():
                        return NAME_TO_TYPE[n]
            raise e

NAME_TO_TYPE: Mapping[str, 'PacketDataType'] = {
    'string': PacketDataType.STRING,
    'bool': PacketDataType.BOOL,
    '::std::vector<bool>': PacketDataType.BOOL_LIST,
    'int': PacketDataType.INT,
    '::std::vector<int>': PacketDataType.INT_LIST,
    'int64': PacketDataType.INT,
    'int64_t': PacketDataType.INT,
    '::std::vector<int64>': PacketDataType.INT_LIST,
    '::std::vector<int64_t>': PacketDataType.INT_LIST,
    'float': PacketDataType.FLOAT,
    '::std::vector<float>': PacketDataType.FLOAT_LIST,
    '::mediapipe::Matrix': PacketDataType.AUDIO,
    '::mediapipe::ImageFrame': PacketDataType.IMAGE_FRAME,
    '::mediapipe::Classification': PacketDataType.PROTO,
    '::mediapipe::ClassificationList': PacketDataType.PROTO,
    '::mediapipe::ClassificationListCollection': PacketDataType.PROTO,
    '::mediapipe::Detection': PacketDataType.PROTO,
    '::mediapipe::DetectionList': PacketDataType.PROTO,
    '::mediapipe::Landmark': PacketDataType.PROTO,
    '::mediapipe::LandmarkList': PacketDataType.PROTO,
    '::mediapipe::LandmarkListCollection': PacketDataType.PROTO,
    '::mediapipe::NormalizedLandmark': PacketDataType.PROTO,
    '::mediapipe::FrameAnnotation': PacketDataType.PROTO,
    '::mediapipe::Trigger': PacketDataType.PROTO,
    '::mediapipe::Rect': PacketDataType.PROTO,
    '::mediapipe::NormalizedRect': PacketDataType.PROTO,
    '::mediapipe::NormalizedLandmarkList': PacketDataType.PROTO,
    '::mediapipe::NormalizedLandmarkListCollection': PacketDataType.PROTO,
    '::mediapipe::Image': PacketDataType.IMAGE,
    '::std::vector<::mediapipe::Image>': PacketDataType.IMAGE_LIST,
    '::std::vector<::mediapipe::Classification>': PacketDataType.PROTO_LIST,
    '::std::vector<::mediapipe::ClassificationList>': PacketDataType.PROTO_LIST,
    '::std::vector<::mediapipe::Detection>': PacketDataType.PROTO_LIST,
    '::std::vector<::mediapipe::DetectionList>': PacketDataType.PROTO_LIST,
    '::std::vector<::mediapipe::Landmark>': PacketDataType.PROTO_LIST,
    '::std::vector<::mediapipe::LandmarkList>': PacketDataType.PROTO_LIST,
    '::std::vector<::mediapipe::NormalizedLandmark>': PacketDataType.PROTO_LIST,
    '::std::vector<::mediapipe::NormalizedLandmarkList>': (
        PacketDataType.PROTO_LIST
    ),
    '::std::vector<::mediapipe::Rect>': PacketDataType.PROTO_LIST,
    '::std::vector<::mediapipe::NormalizedRect>': PacketDataType.PROTO_LIST,
    '::mediapipe::Joint': PacketDataType.PROTO,
    '::mediapipe::JointList': PacketDataType.PROTO,
    '::std::vector<::mediapipe::JointList>': PacketDataType.PROTO_LIST,
    '::mediapipe::Tensor': PacketDataType.TENSOR, #added tensor type
}

# ... (rest of the SolutionBase class remains largely the same, with modifications for MPII) ...

class SolutionBase:
    """The common base class for the high-level MediaPipe Solution APIs.

    Modified to support MPII Human Pose dataset training.
    """
    # ... (rest of the class definition) ...
    def process(
        self, input_data: Union[np.ndarray, Mapping[str, Union[np.ndarray, message.Message, tf.Tensor]]]
    ) -> NamedTuple:
        """Processes a set of RGB image data and output SolutionOutputs.

        Args:
            input_data: Either a single numpy ndarray object representing the solo
                image input of a graph or a mapping from the stream name to the image or
                proto data or tensor data that represents every input streams of a graph.

        Raises:
            NotImplementedError: If input_data contains audio data or a list of proto
                objects.
            RuntimeError: If the underlying graph occurs any error.
            ValueError: If the input image data is not three channel RGB.

        Returns:
            A NamedTuple object that contains the output data of a graph run.
            The field names in the NamedTuple object are mapping to the graph output
            stream names.

        Examples:
            solution = solution_base.SolutionBase(graph_config=hand_landmark_graph)
            results = solution.process(cv2.imread('/tmp/hand0.png')[:, :, ::-1])
            print(results.detection)
            results = solution.process(
                {'video_in' : cv2.imread('/tmp/hand1.png')[:, :, ::-1]})
            print(results.hand_landmarks)
        """
        # ... (rest of the process method, including tensor support) ...
        self._graph_outputs.clear()

        if isinstance(input_data, np.ndarray):
            if self._input_stream_type_info is None:
                raise ValueError('_input_stream_type_info is None in SolutionBase')
            if len(self._input_stream_type_info.keys()) != 1:
                raise ValueError(
                    "Can't process single image input since the graph has more than one"
                    ' input streams.'
                )
            input_dict = {next(iter(self._input_stream_type_info)): input_data}
        else:
            input_dict = input_data

        # Set the timestamp increment to 33333 us to simulate the 30 fps video
        # input.
        self._simulated_timestamp += 33333
        if self._graph is None:
            raise ValueError('_graph is None in SolutionBase')
        for stream_name, data in input_dict.items():
            input_stream_type = self._input_stream_type_info[stream_name]
            if (input_stream_type == PacketDataType.PROTO_LIST or
                input_stream_type == PacketDataType.AUDIO):
                # TODO: Support audio data.
                raise NotImplementedError(
                    f'SolutionBase can only process non-audio and non-proto-list data. '
                    f'{self._input_stream_type_info[stream_name].name} '
                    f'type is not supported yet.')
            elif (input_stream_type == PacketDataType.IMAGE_FRAME or
                  input_stream_type == PacketDataType.IMAGE):
                if data.shape[2] != RGB_CHANNELS:
                    raise ValueError('Input image must contain three channel rgb data.')
                self._graph.add_packet_to_input_stream(
                    stream=stream_name,
                    packet=self._make_packet(input_stream_type,
                                             data).at(self._simulated_timestamp))
            elif input_stream_type == PacketDataType.TENSOR:
                self._graph.add_packet_to_input_stream(
                    stream=stream_name,
                    packet=self._make_packet(input_stream_type, data).at(self._simulated_timestamp)
                )

            else:
                self._graph.add_packet_to_input_stream(
                    stream=stream_name,
                    packet=self._make_packet(input_stream_type,
                                             data).at(self._simulated_timestamp))

        self._graph.wait_until_idle()
        # Create a NamedTuple object where the field names are mapping to the graph
        # output stream names.
        if self._output_stream_type_info is None:
            raise ValueError('_output_stream_type_info is None in SolutionBase')
        solution_outputs = collections.namedtuple(
            'SolutionOutputs', self._output_stream_type_info.keys())
        for stream_name in self._output_stream_type_info.keys():
            if stream_name in self._graph_outputs:
                setattr(
                    solution_outputs, stream_name,
                    self._get_packet_content(self._output_stream_type_info[stream_name],
                                             self._graph_outputs[stream_name]))
            else:
                setattr(solution_outputs, stream_name, None)

        return solution_outputs

    # ... (rest of the class definition) ...

    def _make_packet(self, packet_data_type: PacketDataType,
                    data: Any) -> packet.Packet:
        if (packet_data_type == PacketDataType.IMAGE_FRAME or
            packet_data_type == PacketDataType.IMAGE):
            return getattr(packet_creator, 'create_' + packet_data_type.value)(
                data, image_format=image_frame.ImageFormat.SRGB)

        elif packet_data_type == PacketDataType.TENSOR:
            return packet_creator.create_tensor(data)

        else:
            return getattr(packet_creator, 'create_' + packet_data_type.value)(data)

    def _get_packet_content(self, packet_data_type: PacketDataType,
                            output_packet: packet.Packet) -> Any:
        """Gets packet content from a packet by type.

        Args:
            packet_data_type: The supported packet data type.
            output_packet: The packet to get content from.

        Returns:
            Packet content by packet data type. None to indicate "no output".

        """

        if output_packet.is_empty():
            return None
        if packet_data_type == PacketDataType.STRING:
            return packet_getter.get_str(output_packet)
        elif (packet_data_type == PacketDataType.IMAGE_FRAME or
              packet_data_type == PacketDataType.IMAGE):
            return getattr(packet_getter, 'get_' +
                           packet_data_type.value)(output_packet).numpy_view()
        elif packet_data_type == PacketDataType.TENSOR:
            return packet_getter.get_tensor(output_packet).numpy_view()
        else:
            return getattr(packet_getter, 'get_' + packet_data_type.value)(
                output_packet)

    def __enter__(self):
        """A "with" statement support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Closes all the input sources and the graph."""
        self.close()