-- layer3-spatial-gnn/postgis-clipping.sql (blueprint §7.1)
-- Deterministic Boolean exclusion of wetlands / habitats / urban / floodplains
-- with statutory buffers, plus Euclidean distances to grid & road evacuation.
CREATE EXTENSION IF NOT EXISTS postgis;

WITH unifiedExclusions AS (
    SELECT ST_Union(geom) AS geom FROM (
        SELECT ST_Buffer(geom::geography, 1000)::geometry AS geom FROM protectedWetlands
        UNION ALL
        SELECT ST_Buffer(geom::geography, 500)::geometry  AS geom FROM ecologicalHabitats
        UNION ALL
        SELECT ST_Buffer(geom::geography, 2000)::geometry AS geom FROM urbanMunicipalZones
        UNION ALL
        SELECT ST_Buffer(geom::geography, 250)::geometry  AS geom FROM floodplains100Yr
    ) AS bufferedConstraints
)
INSERT INTO permissibleLandParcels (parcelId, geom, areaSqKm, distSubstationM, distHighwayM)
SELECT
    c.parcelId,
    ST_Multi(ST_Difference(c.geom, e.geom)) AS geom,
    ST_Area(ST_Difference(c.geom, e.geom)::geography) / 1000000.0 AS areaSqKm,
    (SELECT MIN(ST_Distance(c.geom::geography, s.geom::geography)) FROM electricalSubstations s
       WHERE s.voltageKv >= 66),
    (SELECT MIN(ST_Distance(c.geom::geography, r.geom::geography)) FROM highwayAccessRoutes r)
FROM candidateCadastreParcels c
CROSS JOIN unifiedExclusions e
WHERE NOT ST_IsEmpty(ST_Difference(c.geom, e.geom))
  AND (ST_Area(ST_Difference(c.geom, e.geom)::geography) / 1000000.0) >= 0.25;
