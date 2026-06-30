import logging

from aidetector.identity.service import IdentityService
from aidetector.utils.config import (
    Detection,
    DetectorIdentityConfig,
    ImageSet,
)


class IdentityEnricher:
    logger = logging.getLogger(__name__)

    def __init__(self, service: IdentityService, config: DetectorIdentityConfig):
        self.service = service
        self.config = config

    def enrich(
        self,
        source: str,
        detection: Detection,
    ) -> None:
        detection.identities = []

        crop = detection.images.best_crop
        if crop is None:
            self.logger.info("Skipping identity: no crop available")
            return

        self.logger.info(
            "Looking up identity with provider %s for best detector crop",
            self.config.provider,
        )
        lookup_detection = Detection(
            detection.date,
            ImageSet(detection.images.jpg, [crop.clone()]),
            detection.confidence,
        )
        identities = self.service.identify(
            self.config.provider,
            lookup_detection,
            source,
            multiple=self.config.multiple,
        )
        if not identities:
            self.logger.info(
                "Identity provider returned no result for detector crop label %s",
                crop.label,
            )
            return

        primary = identities[0]
        self.logger.info(
            "Identity result: status=%s id=%s similarity=%s",
            primary.status,
            primary.identity_id,
            primary.similarity,
        )
        detection.identities = identities
